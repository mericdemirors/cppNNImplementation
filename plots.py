import seaborn as sns
import matplotlib.pyplot as plt
import sys

train_losses = [float(l) for l in sys.argv[1].split(",")]
valid_losses = [float(l) for l in sys.argv[2].split(",")]
y_test = [float(y) for y in sys.argv[3].split(",")]
test_pred = [float(p) for p in sys.argv[4].split(",")]

# loss plot
plt.plot(range(len(train_losses)), train_losses, label = "train error")
plt.plot(range(len(valid_losses)), valid_losses, label = "validation error")
plt.legend()
plt.show()

y_test.sort();
test_pred.sort();
# pred plot
plt.plot(range(len(y_test)), y_test, label = "true")
plt.plot(range(len(y_test)), test_pred, label = "test predictions")
plt.legend()
plt.show()


f = open("NNlog.txt", "w")
f.close()
f = open("NNlog.txt", "a")
f.write("train_losses:\n")
f.write(str(train_losses))
f.write("\nvalid_losses:\n")
f.write(str(valid_losses))
f.write("\ny_test:\n")
f.write(str(y_test))
f.write("\ntest_pred:\n")
f.write(str(test_pred))
f.close()
