import numpy as np

def GenerateAnchor(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    base_anchor = np.array([1, 1, base_size, base_size]) - 1

    w, h, x_ctr, y_ctr = WH_XYCtr(base_anchor)
    wh_size = w * h
    size_ratios = wh_size / ratios

    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = MakeAnchors(ws, hs, x_ctr, y_ctr)

    anchors = np.vstack([AnchorScales(anchor, scales) for anchor in anchors])

    return anchors


def WH_XYCtr(box):
    w = box[2] - box[0] + 1
    h = box[3] - box[1] + 1
    x_ctr = box[0] + 0.5 * (w - 1)
    y_ctr = box[1] + 0.5 * (h - 1)

    return w, h, x_ctr, y_ctr


def MakeAnchors(Widths, Heights, Xctr, Yctr):
    Widths = Widths[:, np.newaxis]
    Heights = Heights[:, np.newaxis]
    anchors = np.hstack((
        Xctr - 0.5 * (Widths - 1),
        Yctr - 0.5 * (Heights - 1),
        Xctr + 0.5 * (Widths - 1),
        Yctr + 0.5 * (Heights - 1),
    ))

    return anchors


def AnchorScales(anchor, scales):
    w, h, x_ctr, y_ctr = WH_XYCtr(anchor)
    ws = w * scales
    hs = h * scales
    anchors = MakeAnchors(ws, hs, x_ctr, y_ctr)

    return anchors