from src.dataset_registry import DatasetRegistry
from src.datasets.arctic_dataset import ArcticDataset
from src.datasets.arctic_light_dataset import ArcticLightDataset
from src.datasets.h2o_dataset import H2ODataset
from src.datasets.h2o3d_dataset import H2O3DDataset
from src.datasets.dexycb_dataset import DexYCBDataset
from src.datasets.hot3d_dataset import HOT3DDataset
from src.datasets.assembly_dataset import AssemblyDataset
from src.datasets.epic_dataset import EPICDataset
from src.datasets.holo_dataset import HoloDataset
from src.datasets.ego_exo_dataset import EgoExoDataset
from src.datasets.motion_dataset import FixedLengthMotion, VariableLengthMotion, VariableLengthMotion2D

DatasetRegistry.register("arctic_dataset")(ArcticDataset)
DatasetRegistry.register("arctic_light_dataset")(ArcticLightDataset)
DatasetRegistry.register("h2o_dataset")(H2ODataset)
DatasetRegistry.register("h2o3d_dataset")(H2O3DDataset)
DatasetRegistry.register("dexycb_dataset")(DexYCBDataset)
DatasetRegistry.register("hot3d_dataset")(HOT3DDataset)
DatasetRegistry.register("assembly_dataset")(AssemblyDataset)
DatasetRegistry.register("epic_dataset")(EPICDataset)
DatasetRegistry.register("holo_dataset")(HoloDataset)
DatasetRegistry.register("ego_exo_dataset")(EgoExoDataset)
DatasetRegistry.register(["fixed_length_motion", "fixedmotion"])(FixedLengthMotion)
DatasetRegistry.register(["variable_length_motion", "variablemotion"])(VariableLengthMotion)
DatasetRegistry.register(["variable_length_motion_2d", "variablemotion2d"])(VariableLengthMotion2D)
