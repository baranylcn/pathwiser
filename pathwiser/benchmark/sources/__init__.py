from pathwiser.benchmark.sources import abcd, intents, sgd, star

LOADERS = {
    "sgd": sgd.load,
    "banking77": intents.load_banking77,
    "clinc150": intents.load_clinc150,
    "hwu64": intents.load_hwu64,
    "star": star.load,
    "abcd": abcd.load,
}
