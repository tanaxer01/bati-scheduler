import matplotlib.pyplot as plt
import numpy as np
import json
import sys


def train():
    with open("../expe-out/scores.json") as scores_f:
        scores = json.load(scores_f)

    for i in scores.values():
        scores, queue_len, wait, loss = np.array(i["scores"]), i["queue_len"], np.array(i["wait"]), i["loss"]

        scores = scores[ scores != 0 ]
        mean_score = [ scores[:i].mean() for i in range(scores.shape[0]) ]

        mean_wait = wait.mean()
        print(mean_wait)

        mean_loss = [ sum(loss[:i])/i for i in range(1, len(loss)) ]

        plt.figure()
        plt.subplot(4,1,1)
        plt.plot(scores)
        plt.plot(mean_score)

        plt.subplot(4,1,2)
        plt.plot(queue_len)

        plt.subplot(4,1,3)
        plt.ylim([0, 50])
        plt.plot(wait)
        #plt.axhline(y=10, color='r')

        plt.subplot(4,1,4)
        plt.plot(loss)
        plt.show()


def play():
    with open("../expe-out/play_scores.json") as scores_f:
        logs = json.load(scores_f)

    scores, queue_len, wait = np.array(logs["scores"]), logs["queue_len"], np.array(logs["wait"])

    scores = scores[ scores != 0 ]
    mean_score = [ scores[:i].mean() for i in range(scores.shape[0]) ]

    mean_wait = wait.mean()
    print(mean_wait)

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(scores)
    plt.plot(mean_score)

    plt.subplot(3,1,2)
    plt.plot(queue_len)

    plt.subplot(3,1,3)
    #plt.ylim([0, 50])
    plt.plot(wait)
    plt.axhline(y=1, color='r')


    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "train":
        train()
    else:
        print(sys.argv)
        play()





