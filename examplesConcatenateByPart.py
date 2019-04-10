from pickle import Pickler, Unpickler
from hyper import Hyper
import os


def getNames():
    names = []
    for iteration in range(30):
        subnames = []
        for part in range(Hyper.processing):
            filePath = os.path.join(
                Hyper.examples, 'overTrain.examples.part-' + str(part)+'.iter-'+str(iteration))
            subnames.append(filePath)
        names.append(subnames)
    return names


def saveOvertrainingExamples(subnames, iteration):
    overtrainingExamples = []
    for name in subnames:
        with open(name, 'rb') as f:
            examples = Unpickler(f).load()
            for e in examples:
                overtrainingExamples.extend(e)
            f.close()

    filename = os.path.join(
        Hyper.examples, 'overtraining.examples.iter-'+str(iteration))
    with open(filename, 'wb+') as f:
        Pickler(f).dump(overtrainingExamples)
    f.close()


if __name__ == "__main__":
    names = getNames()
    for iteration in range(30):
        saveOvertrainingExamples(names[iteration], iteration)
