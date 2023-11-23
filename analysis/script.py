import matplotlib.pyplot as plt
import pandas as pd


neg_loss = pd.read_csv("loss_neg_reward.csv")
frac_loss = pd.read_csv("loss_fraction_reward.csv")

neg_loss = (neg_loss - neg_loss.mean()) / neg_loss.std()
frac_loss = (frac_loss - frac_loss.mean()) / frac_loss.std()

plt.figure()
plt.plot(neg_loss)
plt.plot(frac_loss)

plt.show()
