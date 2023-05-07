from Network import Network

if __name__ == '__main__':
    network = Network('f8')
    result, weights = network.test('1p')
    print(weights)
