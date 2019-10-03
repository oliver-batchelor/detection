

def count_classes(label, num_classes):

    class_counts = (label + 1).view(-1).bincount(minlength = num_classes + 2)

    return struct(
        ignored  = class_counts[0].item(),
        negative = class_counts[1].item(),
        classes = class_counts[2:],

        positive = class_counts[2:].sum().item(),
        total = label.numel()
    )


def count_instances(label, num_classes):
    return label.view(-1).bincount(minlength = num_classes + 1)



def log_boxes(name, class_names, counts, log):
    assert len(class_names) == counts.classes.size(0)

    class_counts = {"class_{}".format(c):count for c, count in zip(class_names, counts.classes) }

    log.scalars(name + "/boxes",
        struct(ignored = counts.ignored, positive = counts.positive, **class_counts))


def batch_stats(batch):
    assert(batch.dim() == 4 and batch.size(3) == 3)

    batch = batch.float().div_(255)
    flat = batch.view(-1, 3)

    return batch.size(0) * struct(mean=flat.mean(0).cpu(), std=flat.std(0).cpu())


def log_predictions(name, class_names, histograms, log):

    assert len(histograms) == len(class_names)
    totals = reduce(operator.add, histograms)

    if len(class_names)  > 1:
        for i in range(0, len(class_names)):
            class_name = class_names[i]

            log.histogram(name + "/" + class_name + "/positive", histograms[i].positive)
            log.histogram(name + "/" + class_name + "/negative", histograms[i].negative)

    log.histogram(name + "/positive", totals.positive)
    log.histogram(name + "/negative", totals.negative)


def prediction_stats(encoding, prediction, num_bins = 50):

    num_classes = prediction.classification.size(2)
    dist_histogram = torch.LongTensor(2, num_classes, num_bins)

    def class_histogram(i):
        pos_mask = encoding.classification == (i + 1)
        neg_mask = (encoding.classification > 0) & ~pos_mask

        class_pred = prediction.classification.select(2, i)

        return struct (
            positive = Histogram(values = class_pred[pos_mask], range = (0, 1), num_bins = num_bins),
            negative = Histogram(values = class_pred[neg_mask], range = (0, 1), num_bins = num_bins)
        )

    return ZipList(class_histogram(i) for i in range(0, num_classes))



def summarize_stats(results, epoch, globals={}):
    avg = mean_results(results)
    counts = avg.box_counts
    

    print ("image: mean = {}, std = {}".format(str(avg.image.mean), str(avg.image.std)))
    print("instances: {:.2f}, anchors {:.2f}, anchors/instance {:.2f}, positive {:.2f},  ignored {:.2f}, negative {:.2f} "
        .format(avg.boxes, counts.total, counts.positive / avg.boxes, counts.positive, counts.ignored, counts.negative ))

    balances = counts.classes / counts.positive
    print("class balances: {}".format(str(balances.tolist())))    