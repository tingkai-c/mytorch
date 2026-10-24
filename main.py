from mytorch import Value
from mytorch.nn import Linear, Sequential, Sigmoid


def main():
    target = Value(0.427)
    learning_rate = 0.1
    epochs = 3500


    network = Sequential(
        Linear(4,12),
        Sigmoid(),
        Linear(12,12),
        Sigmoid(),
        Linear(12,12),
        Sigmoid(),
        Linear(12,1),
        Sigmoid()
    )

    inputs = [[1,2,3,4],[2,4,6,8],[3,6,9,12],[4,8,12,16],[100,100,100,100]]
    for epoch in range(epochs):

        losses = [(network.forward(inputs[i])[0] - target) ** 2 for i in range(len(inputs))]

        loss = sum(losses)
        print(loss)

        loss.zerograd()
        loss.backward()

        for param in network.parameters():
            param.data -= param.grad * learning_rate

    print([network.forward(inputs[i]) for i in range(len(inputs))])


if __name__ == "__main__":
    main()
