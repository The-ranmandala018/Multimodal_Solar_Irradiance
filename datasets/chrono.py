"""Chronological splitting utilities for Folsom benchmark samples."""


def chronological_year_split(
    samples,
    val_split=0.1,
    train_years=(2014, 2015),
    test_year=2016,
):
    """Split samples by calendar year without random mixing across years."""
    trainval = [
        i
        for i, sample in enumerate(samples)
        if sample[1].year in train_years
    ]

    test = [
        i
        for i, sample in enumerate(samples)
        if sample[1].year == test_year
    ]

    if len(trainval) < 2 or not test:
        raise RuntimeError(
            f"Invalid chronological split: "
            f"train/validation={len(trainval)}, test={len(test)}"
        )

    val_len = max(
        1,
        int(len(trainval) * float(val_split)),
    )
    val_len = min(
        val_len,
        len(trainval) - 1,
    )

    train = trainval[:-val_len]
    val = trainval[-val_len:]

    return train, val, test
