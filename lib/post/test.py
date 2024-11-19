import math
from iglm import IgLM
from anarci import anarci

iglm = IgLM()

sequence = "EVQLVETGGGLVQPGGSLRLSCAASGFTLNSYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSVKGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAKGRAAGTFDSWGQGTLVTVSS"
chain_token = "[HEAVY]"
species_token = "[HUMAN]"

log_likelihood = iglm.log_likelihood(
    sequence,
    chain_token,
    species_token,
)
print(119*log_likelihood)
