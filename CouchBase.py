import os
import pydicom
import numpy as np
from scipy.optimize import curve_fit
import configparser
import logging

logging.basicConfig(level=logging.DEBUG,
                    filename='logs.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


def ReplaceCouchBase():
    config = configparser.ConfigParser()
    config.read('CouchBase.ini')

    tableCTmean = float(config['InsertTableSurface']['TableCTmean'])
    tableCTmeanTol = float(config['InsertTableSurface']['TableCTmeanTolerance'])
    tableCTstd = float(config['InsertTableSurface']['TableCTstd'])
    tableCTstdTol = float(config['InsertTableSurface']['TableCTstdTolerance'])
    dotCTLow = float(config['dotThreshold']['DotCTLow'])
    dotCTHigh = float(config['dotThreshold']['DotCTHigh'])
    CTrows = int(config['Default']['CTRows'])
    CTcols = int(config['Default']['CTCols'])

    OriginalCTDir = 'C:/Users/francisyu.NYPROTON/OneDrive - New York Proton Center/Documents/PelvicPhantomCT'
    flist = os.listdir(OriginalCTDir)
    CTslices = len(flist)

    dataset = pydicom.dcmread(os.path.join(OriginalCTDir, flist[0]))
    pixelSpacing = dataset.PixelSpacing
    sliceThickness = dataset.SliceThickness
    imagePosition = dataset.ImagePositionPatient
    CTNumbers = np.zeros((CTslices, CTrows, CTcols))

    # Read CTs to CTNumbers
    i = 0
    for fname in flist:
        fullname = os.path.join(OriginalCTDir, fname)
        if os.path.isfile(fullname):
            dataset = pydicom.dcmread(fullname)
            CTNumbers[i, :, :] = dataset.pixel_array
            i = i + 1
    logging.info('Total {} CT slices read.'.format(i))

    # Try to find the table
    lBound = CTcols // 2 - int(100.0 / pixelSpacing[0])  # central area, 20cm wide
    rBound = CTcols // 2 + int(100.0 / pixelSpacing[0])
    tableRow = []
    for row in range(CTrows // 2, CTrows):  # search from half of the CT down
        if (abs(np.mean(CTNumbers[:, row, lBound:rBound]) - tableCTmean) < tableCTmeanTol)and\
                (abs(np.std(CTNumbers[:, row, lBound:rBound]) - tableCTstd) < tableCTstdTol):
            tableRow.append(row)
    firstRow = tableRow[0]
    logging.debug('Find all row meet mean/std criteria as {}'.format(tableRow))
    logging.info('Find table in row {}'.format(firstRow))

    # On the table surface find the dots
    dots_idx = np.argwhere(
        (CTNumbers[:, firstRow, lBound:rBound] > dotCTLow) & (CTNumbers[:, firstRow, lBound:rBound] < dotCTHigh))
    logging.debug('Find {} dots in the table surface meet dotCTHigh and dotCTLow'.format(len(dots_idx)))
    dots_checked = CheckPoints(CTNumbers, dots_idx, firstRow, lBound, sliceThickness, pixelSpacing)
    logging.info('Total {} dots pass the dots Gaussian fit check'.format(len(dots_checked)))
    logging.debug('Dots list: {}'.format(dots_checked))

    # Get the origin dot(0)
    dot_origin = GetDotOrigin(dots_checked, sliceThickness, pixelSpacing)
    logging.info('Dot origin is {}'.format(dot_origin))
    dot_origin_abs = []
    for d in dot_origin:  # Convert dot_origin to absolute dot origin
        dot_origin_abs.append([d[1] * pixelSpacing[0] + imagePosition[0],
                              firstRow * pixelSpacing[1] + imagePosition[1],
                              imagePosition[2] - d[0] * sliceThickness - d[2]])
    logging.info('Dot absolute origin is {}'.format(dot_origin_abs))

    # Register CT with couch base on dot origin
    RegisteredCTNumbers = RegistrationCouch(CTNumbers, dot_origin_abs[-1], sliceThickness, pixelSpacing, imagePosition)

    i = 0
    newDir = os.path.join(OriginalCTDir, 'new')  # Add a 'new' folder in current folder
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    for fname in flist:
        fullname = os.path.join(OriginalCTDir, fname)
        if os.path.isfile(fullname):
            dataset = pydicom.dcmread(fullname)
            for j in range(RegisteredCTNumbers.shape[1]):
                for k in range(RegisteredCTNumbers.shape[2]):
                    dataset.pixel_array[j, k] = RegisteredCTNumbers[i, j, k]
            dataset.PixelData = dataset.pixel_array.tobytes()
            dataset.save_as(os.path.join(newDir, fname))
            i += 1
    logging.info('Total {} files are written.'.format(i))
    logging.info('FINISH!')

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def CheckPoints(CTNumbers, dots_idx, row, lBound, sliceThickness, pixelSpacing):
    # Use Gaussian fit to check if the point is a 'real' point
    config = configparser.ConfigParser()
    config.read('CouchBase.ini')

    results = []
    cutSigmaX = pixelSpacing[0] * float(config['Gaussian']['Sigma']);
    cutSigmaY = sliceThickness * float(config['Gaussian']['Sigma']);
    cutMean = float(config['Gaussian']['Mean'])
    cutAmp = float(config['Gaussian']['Amp'])
    initSigma = float(config['Gaussian']['InitSigma'])
    initMean = float(config['Gaussian']['InitMean'])
    initAmp = float(config['Gaussian']['InitAmp'])
    for i in range(len(dots_idx)):
        CTslice = dots_idx[i, 0]
        col = dots_idx[i, 1] + lBound
        axis = np.linspace(-2, 2)
        init = [initAmp, initMean, initSigma]

        profileX = CTNumbers[CTslice, row, col - 2: col + 3]
        profileY = CTNumbers[CTslice - 2: CTslice + 3, row, col]

        pX = np.interp(axis, [-2, -1, 0, 1, 2], profileX)
        pY = np.interp(axis, [-2, -1, 0, 1, 2], profileY)
        best_valsX, covarX = curve_fit(gauss_function, axis, pX, p0=init)
        best_valsY, covarY = curve_fit(gauss_function, axis, pY, p0=init)
        if best_valsX[0] > cutAmp and best_valsX[1] < cutMean and best_valsX[2] < cutSigmaX\
                and best_valsY[0] > cutAmp and best_valsY[1] < cutMean and best_valsY[2] < cutSigmaY:
            results.append([dots_idx[i, 0], dots_idx[i, 1] + lBound])
    return results


def GetDotOrigin(dots_checked, sliceThickness, pixelSpacing):
    # Find dot origin by using different dots distance relations
    # distance from origin is 140mm
    # distance in same slice is 10mm, 20mm, 30mm and so on
    result = []
    x = list(list(zip(*dots_checked))[0])
    y = list(list(zip(*dots_checked))[1])
    diffx = [(x[i] - x[i - 1]) for i in range(1, len(x))]
    diffy = [(y[i] - y[i - 1]) for i in range(1, len(y))]
    ordr = 1
    for i in range(len(diffx)):
        if abs(diffx[i]) == 1 or abs(diffy[i]) == 1:
            del x[i + ordr]
            del y[i + ordr]
            ordr -= 1
    for i in range(1, len(x)):
        if x[i] == x[i - 1]:
            dist_mm = (y[i] - y[i - 1]) * pixelSpacing[0]
            if abs(round(dist_mm / 10) * 10 - dist_mm) < 0.75:
                distanceX = round(dist_mm / 10) * 10
                distanceY = distanceX / 10 * 140
                result.append([x[i], y[i], distanceY])
    return result


def RegistrationCouch(CTNumbers, dot_origin_abs, sliceThickness, pixelSpacing, position):
    # Register couch to CT using couch model
    config = configparser.ConfigParser()
    config.read('CouchBase.ini')

    couchFile = config['CouchModel']['CouchFile']
    arrName = config['CouchModel']['ArrayName']
    couchModel = np.load(couchFile)[arrName]

    couchPixelSpacing = float(config['CouchModel']['CouchPixelSpacing'])
    couchSliceThickness = float(config['CouchModel']['CouchSliceThickness'])
    couchOrigin = list(map(int, config['CouchModel']['CouchOrigin'].split(',')))
    baseRow = int(config['CouchModel']['BaseRow'])

    baseDist = (baseRow - couchOrigin[1]) * couchPixelSpacing
    destRow = round((dot_origin_abs[1] - position[1] + baseDist) / pixelSpacing[0])
    colStart = round((dot_origin_abs[0] - position[0] - 250) / pixelSpacing[1])
    colEnd = round((dot_origin_abs[0] - position[0] + 250) / pixelSpacing[1])

    for CTslice in range(CTNumbers.shape[0]):
        for row in range(destRow, CTNumbers.shape[1]):
            for col in range(int(colStart), int(colEnd) + 1):
                distToOrigin = [col * pixelSpacing[0] - (dot_origin_abs[0] - position[0]),
                                row * pixelSpacing[1] - (dot_origin_abs[1] - position[1]),
                                (position[2] - dot_origin_abs[2]) - CTslice * sliceThickness]
                couchGrid = [couchOrigin[0] - round(distToOrigin[2] / couchSliceThickness),
                             couchOrigin[1] + round(distToOrigin[1] / couchPixelSpacing),
                             couchOrigin[2] + round(distToOrigin[0] / couchPixelSpacing)]
                CTNumbers[CTslice, row, col] = couchModel[int(couchGrid[0]), int(couchGrid[1]), int(couchGrid[2])]
    return CTNumbers


if __name__ == "__main__":
    logging.info('START...')
    ReplaceCouchBase()
