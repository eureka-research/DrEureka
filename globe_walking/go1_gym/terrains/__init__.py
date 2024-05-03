from .terrain import Terrain
from .tm_box_terrain import TMBoxTerrain
from .box_terrain import BoxTerrain
from .heightfield_terrain import HeightfieldTerrain
from .trimesh_terrain import TrimeshTerrain
from .ground_plane_terrain import GroundPlaneTerrain

ALL_TERRAINS = {    "boxes_tm": TMBoxTerrain,
                    "boxes": BoxTerrain,
                    "heightfield": HeightfieldTerrain,
                    "trimesh": TrimeshTerrain,
                    "plane": GroundPlaneTerrain
                    }