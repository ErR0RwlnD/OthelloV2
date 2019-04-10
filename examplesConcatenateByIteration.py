from pickle import Pickler, Unpickler
from hyper import Hyper
import os


def getNames():
    names = []
    for part in range(Hyper.processing):
        subnames = []
        for iteration in range(30):
            filePath = os.path.join(
                Hyper.examples, 'overTrain.examples.part-' + str(part)+'.iter-'+str(iteration))
            subnames.append(filePath)
        names.append(subnames)
    return names


def saveOvertrainingExamples(subnames, part):
    overtrainingExamples = []
    for name in subnames:
        with open(name, 'rb') as f:
            examples = Unpickler(f).load()
            for e in examples:
                overtrainingExamples.extend(e)
            f.close()

    filename = os.path.join(
        Hyper.examples, 'overtraining.examples.part-'+str(part))
    with open(filename, 'wb+') as f:
        Pickler(f).dump(overtrainingExamples)
    f.close()


if __name__ == "__main__":
    names = getNames()
    for part in range(Hyper.processing):
        saveOvertrainingExamples(names[part], part)
