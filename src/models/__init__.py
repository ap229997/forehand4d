from src.registry import ModelRegistry

# Register Motion Diffusion models
from src.models.mdm.wrapper import MotionDiffusionWrapper
ModelRegistry.register(["mdm", "mdm_light"])(MotionDiffusionWrapper)

# Register Motion Diffusion Hybrid models
from src.models.mdm_hybrid.wrapper import MotionDiffusionHybridWrapper
ModelRegistry.register(["mdm_hybrid", "mdm_hybrid_light"])(MotionDiffusionHybridWrapper)

# Register Motion Feed Forward models
from src.models.mdm_ff.wrapper import MotionFeedForwardWrapper
ModelRegistry.register(["mdm_ff", "mdm_ff_light"])(MotionFeedForwardWrapper)

# Register Latent Action models
from src.models.latentact.wrapper import LatentActWrapper
ModelRegistry.register(["latentact", "latentact_light"])(LatentActWrapper)
