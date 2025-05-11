
from torch.distributions import Beta
beta_dist = Beta(1.5, 1)
cnt = 0
cnt2 = 0
for i in range(1000):
    # groot
    sample = beta_dist.sample([1])
    sample = (0.999 - sample) / 0.999

    # pi0

    if sample < 0.5:
        # print(sample)
        cnt += 1

print(cnt)