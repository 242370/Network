from Network import Network

if __name__ == '__main__':
    network = Network('f10')
    mess, ref, result, weights = network.test('1p')
    network.error(mess, ref, result)
    print(result)
