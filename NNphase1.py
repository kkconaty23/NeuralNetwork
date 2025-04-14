import sys

def main():
    # x values
    x0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
          11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # x1 is not used, but keeping it like original format
    x1 = [0 for _ in range(20)]  # Dummy second feature

    # Actual y values for y = 3x + 2.3
    y = [3 * x + 2.3 for x in x0]

    # starting weights
    w0 = 0.1
    w1 = 0.0  
    b = 0.0

    
    for i in range(100000):
        loss = 0
        for j in range(len(y)):
            a = w0 * x0[j] + w1 * x1[j] + b
            loss += 0.5 * (y[j] - a) ** 2

            dw0 = -(y[j] - a) * x0[j]
            dw1 = -(y[j] - a) * x1[j]
            db = -(y[j] - a)

            #weights
            w0 = w0 - 0.001 * dw0
            w1 = w1 - 0.001 * dw1
            b = b - 0.001 * db

        if i % 5000 == 0:
            print("epoch", i, "loss:", loss)

    print("w0 (should be ~3):", w0)
    print("b (should be ~2.3):", b)

    # Test predictions
    px0 = 5.5
    px1 = 0.0  # dummy
    output = w0 * px0 + w1 * px1 + b
    print("output for (", px0, ",", px1, ") =", output)

    px0 = 12.1
    px1 = 0.0
    output = w0 * px0 + w1 * px1 + b
    print("output for (", px0, ",", px1, ") =", output)

if __name__ == "__main__":
    sys.exit(main())
