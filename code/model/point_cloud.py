import torch
import torch.nn as nn

class PointCloud(nn.Module):
    def __init__(
        self,
        n_init_points,
        # init_points,
        max_points=131072,
        init_radius=0.5,
        radius_factor=0.15
    ):
        super(PointCloud, self).__init__()
        self.radius_factor = radius_factor
        self.max_points = max_points
        self.init_radius = init_radius
        self.init(n_init_points)

    def init(self, n_init_points):
        print("current point number: ", n_init_points)
        # initialize sphere
        init_points = torch.rand(n_init_points, 3) * 2.0 - 1.0

        init_normals = nn.functional.normalize(init_points, dim=1)
        init_points = init_normals * self.init_radius
        self.register_parameter("points", nn.Parameter(init_points))

    def prune(self, visible_points):
        """Prune not rendered points"""
        self.points = nn.Parameter(self.points.data[visible_points])
        print(
            "Pruning points, original: {}, new: {}".format(
                len(visible_points), sum(visible_points)
            )
        )

    def upsample_points(self, new_points):
        self.points = nn.Parameter(torch.cat([self.points, new_points], dim=0))


    def upsample_400_points(self, new_points):
        num_points = self.points.shape[0]
        num_upsample = 400 - num_points
        if num_upsample == 0:
            self.points = nn.Parameter(self.points)
        else:
            rnd_idx = torch.randint(0, new_points.shape[0], (num_upsample,))
            upsample_point = new_points[rnd_idx, :]
            self.points = nn.Parameter(torch.cat([self.points, upsample_point], dim=0))

    def upsample_800_points(self, new_points):
        num_points = self.points.shape[0]
        num_upsample = 800 - num_points
        if num_upsample == 0:
            self.points = nn.Parameter(self.points)
        else:
            rnd_idx = torch.randint(0, new_points.shape[0], (num_upsample,))
            upsample_point = new_points[rnd_idx, :]
            self.points = nn.Parameter(torch.cat([self.points, upsample_point], dim=0))

    def upsample_1600_points(self, new_points):
        num_points = self.points.shape[0]
        num_upsample = 1600 - num_points
        if num_upsample == 0:
            self.points = nn.Parameter(self.points)
        else:
            rnd_idx = torch.randint(0, new_points.shape[0], (num_upsample,))
            upsample_point = new_points[rnd_idx, :]
            self.points = nn.Parameter(torch.cat([self.points, upsample_point], dim=0))

    def upsample_3200_points(self, new_points):
        num_points = self.points.shape[0]
        num_upsample = 3200 - num_points
        if num_upsample == 0:
            self.points = nn.Parameter(self.points)
        else:
            rnd_idx = torch.randint(0, new_points.shape[0], (num_upsample,))
            upsample_point = new_points[rnd_idx, :]
            self.points = nn.Parameter(torch.cat([self.points, upsample_point], dim=0))

    def upsample_6400_points(self, new_points):
        num_points = self.points.shape[0]
        num_upsample = 6400 - num_points
        if num_upsample == 0:
            self.points = nn.Parameter(self.points)
        else:
            rnd_idx = torch.randint(0, new_points.shape[0], (num_upsample,))
            upsample_point = new_points[rnd_idx, :]
            self.points = nn.Parameter(torch.cat([self.points, upsample_point], dim=0))

    def upsample_10000_points(self, new_points):
        num_points = self.points.shape[0]
        num_upsample = 10000 - num_points
        if num_upsample == 0:
            self.points = nn.Parameter(self.points)
        else:
            rnd_idx = torch.randint(0, new_points.shape[0], (num_upsample,))
            upsample_point = new_points[rnd_idx, :]
            self.points = nn.Parameter(torch.cat([self.points, upsample_point], dim=0))

    def upsample_20000_points(self, new_points):
        num_points = self.points.shape[0]
        num_upsample = 20000 - num_points
        if num_upsample == 0:
            self.points = nn.Parameter(self.points)
        else:
            rnd_idx = torch.randint(0, new_points.shape[0], (num_upsample,))
            upsample_point = new_points[rnd_idx, :]
            self.points = nn.Parameter(torch.cat([self.points, upsample_point], dim=0))

    def upsample_40000_points(self, new_points):
        num_points = self.points.shape[0]
        num_upsample = 40000 - num_points
        if num_upsample == 0:
            self.points = nn.Parameter(self.points)
        else:
            rnd_idx = torch.randint(0, new_points.shape[0], (num_upsample,))
            upsample_point = new_points[rnd_idx, :]
            self.points = nn.Parameter(torch.cat([self.points, upsample_point], dim=0))

    def upsample_80000_points(self, new_points):
        num_points = self.points.shape[0]
        num_upsample = 80000 - num_points
        if num_upsample == 0:
            self.points = nn.Parameter(self.points)
        else:
            rnd_idx = torch.randint(0, new_points.shape[0], (num_upsample,))
            upsample_point = new_points[rnd_idx, :]
            self.points = nn.Parameter(torch.cat([self.points, upsample_point], dim=0))

    def upsample_100000_points(self, new_points):
        num_points = self.points.shape[0]
        num_upsample = 100000 - num_points
        if num_upsample == 0:
            self.points = nn.Parameter(self.points)
        else:
            rnd_idx = torch.randint(0, new_points.shape[0], (num_upsample,))
            upsample_point = new_points[rnd_idx, :]
            self.points = nn.Parameter(torch.cat([self.points, upsample_point], dim=0))





