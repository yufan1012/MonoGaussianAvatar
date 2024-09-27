import os
from pyhocon import ConfigFactory
import torch

import sys
sys.path.append('./')

import utils.general as utils
import utils.plots as plt
import math
import wandb
from functools import partial
from model.monogaussian_avatar_model import MonogaussianAvatar
from model.loss import Loss
import trimesh
print = partial(print, flush=True)


class TrainRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.max_batch = self.conf.get_int('train.max_batch', default='8')
        self.batch_size = min(int(self.conf.get_int('train.max_points_training') / self.conf.get_int('model.point_cloud.n_init_points')), self.max_batch)
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.optimize_expression = self.conf.get_bool('train.optimize_expression')
        self.optimize_pose = self.conf.get_bool('train.optimize_camera')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        os.environ['WANDB_DIR'] = os.path.join(self.exps_folder_name)
        wandb.init(project=kwargs['wandb_workspace'], name=self.subject + '_' + self.methodname, config=self.conf, tags=kwargs['wandb_tags'])

        self.optimize_inputs = self.optimize_expression or self.optimize_pose
        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        if kwargs['is_continue']:
            if kwargs['load_path'] != '':
                load_path = kwargs['load_path']
            else:
                load_path = self.train_dir
            if os.path.exists(os.path.join(load_path)):
                is_continue = True
            else:
                is_continue = False
        else:
            is_continue = False

        utils.mkdir_ifnotexists(self.train_dir)
        utils.mkdir_ifnotexists(self.eval_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.train_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        if self.optimize_inputs:
            self.optimizer_inputs_subdir = "OptimizerInputs"
            self.input_params_subdir = "InputParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.input_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.train_dir, 'runconf.conf')))
        self.file_backup()

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.use_background = self.conf.get_bool('dataset.use_background', default=False)

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                          subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                          json_name=self.conf.get_string('dataset.json_name'),
                                                                                          use_mean_expression=self.conf.get_bool('dataset.use_mean_expression', default=False),
                                                                                          use_var_expression=self.conf.get_bool('dataset.use_var_expression', default=False),
                                                                                          use_background=self.use_background,
                                                                                          is_eval=False,
                                                                                          **self.conf.get_config('dataset.train'))

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         use_background=self.use_background,
                                                                                         is_eval=True,
                                                                                         **self.conf.get_config('dataset.test'))
        print('Finish loading data ...')

        self.model = MonogaussianAvatar(conf=self.conf.get_config('model'),
                                        shape_params=self.train_dataset.shape_params,
                                        img_res=self.train_dataset.img_res,
                                        canonical_expression=self.train_dataset.mean_expression,
                                        canonical_pose=self.conf.get_float('dataset.canonical_pose', default=0.2),
                                        use_background=self.use_background)
        self._init_dataloader()
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = Loss(**self.conf.get_config('loss'), var_expression=self.train_dataset.var_expression)

        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam([
            {'params': list(self.model.parameters())},
        ], lr=self.lr)
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)
        self.upsample_freq = self.conf.get_int('train.upsample_freq', default=5)
        # settings for input parameter optimization
        if self.optimize_inputs:
            num_training_frames = len(self.train_dataset)
            param = []
            if self.optimize_expression:
                init_expression = torch.cat((self.train_dataset.data["expressions"], torch.randn(self.train_dataset.data["expressions"].shape[0], max(self.model.deformer_network.num_exp - 50, 0)).float()), dim=1)
                # print(11111)
                # print(init_expression.size())
                # print(num_training_frames)
                self.expression = torch.nn.Embedding(num_training_frames, self.model.deformer_network.num_exp, _weight=init_expression, sparse=True).cuda()
                param += list(self.expression.parameters())

            if self.optimize_pose:
                self.flame_pose = torch.nn.Embedding(num_training_frames, 15, _weight=self.train_dataset.data["flame_pose"], sparse=True).cuda()
                self.camera_pose = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.data["world_mats"][:, :3, 3], sparse=True).cuda()
                param += list(self.flame_pose.parameters()) + list(self.camera_pose.parameters())
            self.optimizer_cam = torch.optim.SparseAdam(param, self.conf.get_float('train.learning_rate_cam'))

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(load_path, 'checkpoints')
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.start_epoch = saved_model_state['epoch']
            n_points = saved_model_state["model_state_dict"]['pc.points'].shape[0]
            # n_points = con_points.shape[0]
            # batch_size = min(int(self.conf.get_int('train.max_points_training') / n_points), self.max_batch)
            batch_size = min(int(self.conf.get_int('train.max_points_training') / (n_points)), self.max_batch)
            if self.batch_size != batch_size:
                self.batch_size = batch_size
                self._init_dataloader()
            # self.model.pc.init(n_points, con_points)
            self.model.pc.init(n_points)
            self.model.pc = self.model.pc.cuda()

            self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)

            self.model.radius = saved_model_state['radius']

            self.optimizer = torch.optim.Adam([
                {'params': list(self.model.parameters())},
            ], lr=self.lr)

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.optimize_inputs:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_inputs_subdir, str(kwargs['checkpoint']) + ".pth"))
                try:
                    self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])
                except:
                    print("input and camera optimizer parameter group doesn't match")
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.input_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                try:
                    if self.optimize_expression:
                        self.expression.load_state_dict(data["expression_state_dict"])
                    if self.optimize_pose:
                        self.flame_pose.load_state_dict(data["flame_pose_state_dict"])
                        self.camera_pose.load_state_dict(data["camera_pose_state_dict"])
                except:
                    print("expression or pose parameter group doesn't match")

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=4,
                                                            )
        self.n_batches = len(self.train_dataloader)
        self.img_res = self.plot_dataset.img_res
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.save_freq = self.conf.get_int('train.save_freq', default=1)

        self.GT_lbs_milestones = self.conf.get_list('train.GT_lbs_milestones', default=[])
        self.GT_lbs_factor = self.conf.get_float('train.GT_lbs_factor', default=0.5)
        for acc in self.GT_lbs_milestones:
            if self.start_epoch > acc:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
        # if len(self.GT_lbs_milestones) > 0 and self.start_epoch >= self.GT_lbs_milestones[-1]:
        #    self.loss.lbs_weight = 0.

    def _init_dataloader(self):
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=4,
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=10,
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn,
                                                           num_workers=4,
                                                           )
        self.n_batches = len(self.train_dataloader)
    def save_checkpoints(self, epoch, only_latest=False):
        if not only_latest:
            torch.save(
                {"epoch": epoch, "radius": self.model.radius,
                 "model_state_dict": self.model.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
                os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))

        torch.save(
            {"epoch": epoch, "radius": self.model.radius,
                 "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        if self.optimize_inputs:
            if not only_latest:
                torch.save(
                    {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                    os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, "latest.pth"))\

            dict_to_save = {}
            dict_to_save["epoch"] = epoch
            if self.optimize_expression:
                dict_to_save["expression_state_dict"] = self.expression.state_dict()
            if self.optimize_pose:
                dict_to_save["flame_pose_state_dict"] = self.flame_pose.state_dict()
                dict_to_save["camera_pose_state_dict"] = self.camera_pose.state_dict()
            if not only_latest:
                torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, str(epoch) + ".pth"))
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, "latest.pth"))

    def upsample_points(self, epoch):
        current_radius = self.model.radius
        # current_prune_thresh_factor = self.model.prune_thresh_factor
        points = self.model.pc.points.data
        num_p = points.shape[0]
        if epoch <= 100:
            noise = (torch.rand(*points.shape).cuda() - 0.5) * current_radius
        else:
            noise = (torch.rand(*points.shape).cuda() - 0.5) * 0.004
        # noise = (torch.rand(*points.shape).cuda() - 0.5) * scale[0]
        new_points = noise + points
        # self.model.pc.upsample_80000_points(new_points)
        # if num_p <= 800:
        #     self.model.pc.upsample_points(new_points)
        # else:
        #     self.model.pc.upsample_80000_points(new_points)
        # print(points.size())
        # print(new_points.size())
        if epoch < 5:
            self.model.pc.upsample_400_points(new_points)
        elif 5 <= epoch < 10:
            self.model.pc.upsample_800_points(new_points)
        elif 10 <= epoch < 15:
            self.model.pc.upsample_1600_points(new_points)
        elif 15 <= epoch < 20:
            self.model.pc.upsample_3200_points(new_points)
        elif 20 <= epoch < 25:
            self.model.pc.upsample_6400_points(new_points)
        elif 25 <= epoch < 30:
            self.model.pc.upsample_10000_points(new_points)
        elif 30 <= epoch < 40:
            self.model.pc.upsample_20000_points(new_points)
        elif 40 <= epoch < 50:
            self.model.pc.upsample_40000_points(new_points)
        elif 50 <= epoch < 60:
            self.model.pc.upsample_80000_points(new_points)
        elif epoch >= 60:
            self.model.pc.upsample_100000_points(new_points)
        if epoch == 5:
            self.model.radius = 0.75 * current_radius
        elif epoch == 10:
            self.model.radius = 0.75 * current_radius
        elif epoch == 15:
            self.model.radius = 0.75 * current_radius
        elif epoch == 20:
            self.model.radius = 0.75 * current_radius
        elif epoch == 25:
            self.model.radius = 0.75 * current_radius
        elif epoch == 30:
            self.model.radius = 0.75 * current_radius
        elif epoch == 40:
            self.model.radius = 0.75 * current_radius
        elif epoch == 50:
            self.model.radius = 0.75 * current_radius
        elif epoch == 60:
            self.model.radius = 0.9 * current_radius
        elif epoch > 60 and epoch % 5 == 0:
            self.model.radius = 0.75 * current_radius
        if epoch >= 100:
            print("old radius: {}, new radius: {}, sample radius: {}".format(current_radius, self.model.radius, 0.004))
        else:
            print("old radius: {}, new radius: {}, sample radius: {}".format(current_radius, self.model.radius, current_radius))
        print("old points: {}, new points: {}".format(num_p,
                                                      self.model.pc.points.data.shape[0]))
        self.optimizer = torch.optim.Adam([
            {'params': list(self.model.parameters())},
        ], lr=self.lr)

    def file_backup(self):
        from shutil import copyfile
        dir_lis = ['./model', './scripts', './utils', './flame', './datasets']
        os.makedirs(os.path.join(self.train_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.train_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
    def psnr(self,img1, img2):
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def run(self):
        acc_loss = {}
        acc_PSNR = {}
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        for epoch in range(self.start_epoch, self.nepochs + 1):
            if epoch in self.GT_lbs_milestones:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor

            if epoch % (self.save_freq * 5) == 0 and epoch != self.start_epoch:
                self.save_checkpoints(epoch)
            else:
                if epoch % self.save_freq == 0 and (epoch != self.start_epoch or self.start_epoch == 0):
                    self.save_checkpoints(epoch, only_latest=True)
            if (epoch % self.plot_freq == 0 and epoch < 5) or (epoch % self.plot_freq == 0):
                self.model.eval()
                if self.optimize_inputs:
                    if self.optimize_expression:
                        self.expression.eval()
                    if self.optimize_pose:
                        self.flame_pose.eval()
                        self.camera_pose.eval()
                eval_iterator = iter(self.plot_dataloader)
                start_time.record()
                for batch_index in range(len(self.plot_dataloader)):
                    indices, model_input, ground_truth = next(eval_iterator)
                    for k, v in model_input.items():
                        try:
                            model_input[k] = v.cuda()
                        except:
                            model_input[k] = v
                    for k, v in ground_truth.items():
                        try:
                            ground_truth[k] = v.cuda()
                        except:
                            ground_truth[k] = v

                    model_outputs = self.model(model_input)
                    for k, v in model_outputs.items():
                        try:
                            model_outputs[k] = v.detach()
                        except:
                            model_outputs[k] = v
                    plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_'+str(epoch)) for i in range(len(model_input['sub_dir']))]
                    img_names = model_input['img_name'][:, 0].cpu().numpy()
                    print("Plotting images: {}".format(img_names))
                    utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))

                    plt.plot(img_names,
                             model_outputs,
                             ground_truth,
                             plot_dir,
                             epoch,
                             self.img_res,
                             is_eval=False,
                             first=(batch_index==0),
                             )

                    if batch_index == 0:
                        # pc
                        vertices = self.model.pc.points.data
                        vertices = vertices.detach().cpu().numpy()
                        # print(vertices)
                        mesh_colored = trimesh.points.PointCloud(vertices=vertices)
                        exp_dir = os.path.join(self.eval_dir, model_input['sub_dir'][0])
                        mesh_colored.export(exp_dir + '/rendering/{}_mesh.ply'.format(epoch))

                        # canonical points
                        can_vertices = model_outputs['canonical_points'].detach().cpu().numpy()
                        can_mesh_colored = trimesh.points.PointCloud(vertices=can_vertices)
                        exp_dir = os.path.join(self.eval_dir, model_input['sub_dir'][0])
                        can_mesh_colored.export(exp_dir + '/rendering/{}_can_mesh.ply'.format(epoch))

                    # # del model_outputs, ground_truth, vertices, mesh_colored
                    del model_outputs, ground_truth
                end_time.record()
                torch.cuda.synchronize()
                print("Plot time per image: {} ms".format(start_time.elapsed_time(end_time) / len(self.plot_dataset)))
                self.model.train()
                if self.optimize_inputs:
                    if self.optimize_expression:
                        self.expression.train()
                    if self.optimize_pose:
                        self.flame_pose.train()
                        self.camera_pose.train()

            start_time.record()

            # Prunning
            if epoch != self.start_epoch and epoch % self.upsample_freq == 0:
                self.model.pc.prune(self.model.visible_points)
                self.optimizer = torch.optim.Adam([
                    {'params': list(self.model.parameters())},
                ], lr=self.lr)
            # Upsampling
            if epoch % self.upsample_freq == 0:
                if epoch != 0:
                    self.upsample_points(epoch)
                    batch_size = min(
                        int(self.conf.get_int('train.max_points_training') / (self.model.pc.points.shape[0])),
                        self.max_batch)
                    if batch_size != self.batch_size:
                        self.batch_size = batch_size
                        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                                            batch_size=self.batch_size,
                                                                            shuffle=True,
                                                                            collate_fn=self.train_dataset.collate_fn,
                                                                            num_workers=4,
                                                                            )
                        self.n_batches = len(self.train_dataloader)

            # re-init visible point tensor each epoch
            self.model.visible_points = torch.zeros(self.model.pc.points.shape[0]).bool().cuda()

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                for k, v in model_input.items():
                    try:
                        model_input[k] = v.cuda()
                    except:
                        model_input[k] = v
                for k, v in ground_truth.items():
                    try:
                        ground_truth[k] = v.cuda()
                    except:
                        ground_truth[k] = v

                if self.optimize_inputs:
                    if self.optimize_expression:
                        model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                    if self.optimize_pose:
                        model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                        model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)

                model_outputs = self.model(model_input)

                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']

                self.optimizer.zero_grad()
                if self.optimize_inputs and epoch > 10:
                    self.optimizer_cam.zero_grad()

                loss.backward()
                self.optimizer.step()
                if self.optimize_inputs and epoch > 10:
                    self.optimizer_cam.step()

                for k, v in loss_output.items():
                    loss_output[k] = v.detach().item()
                    if k not in acc_loss:
                        acc_loss[k] = [v]
                    else:
                        acc_loss[k].append(v)

                acc_loss['visible_percentage'] = (torch.sum(self.model.visible_points)/self.model.pc.points.shape[0]).unsqueeze(0)
                if data_index % 50 == 0:
                    for k, v in acc_loss.items():
                        acc_loss[k] = sum(v) / len(v)
                    print_str = '{0} [{1}] ({2}/{3}): '.format(self.methodname, epoch, data_index, self.n_batches)
                    for k, v in acc_loss.items():
                        print_str += '{}: {:.3g} '.format(k, v)
                    print(print_str)
                    acc_loss['num_points'] = self.model.pc.points.shape[0]
                    acc_loss['radius'] = self.model.radius

                    acc_loss['lr'] = self.scheduler.get_last_lr()[0]
                    # # print batch size
                    acc_loss['batch_size'] = self.batch_size
                    wandb.log(acc_loss, step=epoch * len(self.train_dataset) + data_index * self.batch_size)
                    acc_loss = {}

            self.scheduler.step()
            end_time.record()
            torch.cuda.synchronize()
            wandb.log({"timing_epoch": start_time.elapsed_time(end_time)}, step=(epoch+1) * len(self.train_dataset))
            print("Epoch time: {} s".format(start_time.elapsed_time(end_time)/1000))
        self.save_checkpoints(self.nepochs + 1)



