from .continuous.bc import BCAgent
from .continuous.gc_bc import GCBCAgent
from .continuous.gc_ddpm_bc import GCDDPMBCAgent
from .continuous.gc_iql import GCIQLAgent
from .continuous.iql import IQLAgent
from .continuous.lc_bc import LCBCAgent
from .continuous.stable_contrastive_rl import StableContrastiveRLAgent
from .continuous.stable_contrastive_rl_vf import StableContrastiveRLVfAgent
from .continuous.gc_discriminator import GCDiscriminatorAgent
from .continuous.lcgc_progress_vf import LCGCProgressVFAgent

agents = {
    "gc_bc": GCBCAgent,
    "lc_bc": LCBCAgent,
    "gc_iql": GCIQLAgent,
    "gc_ddpm_bc": GCDDPMBCAgent,
    "bc": BCAgent,
    "iql": IQLAgent,
    "stable_contrastive_rl": StableContrastiveRLAgent,
    "stable_contrastive_rl_vf": StableContrastiveRLVfAgent,
    "gc_discriminator": GCDiscriminatorAgent,
    "lcgc_progress_vf": LCGCProgressVFAgent,
}
 