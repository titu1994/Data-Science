from collections import defaultdict

def readCSVFile(fn):
    return open(fn, "r", encoding="utf8", newline="")

def writeCSVFile(fn):
    return open(fn, "w", encoding="utf8", newline="")

def parseRow(inputRow, parsers):
    return [tryIfNone(parser)(value) if parser is not None else value
            for value, parser in zip(inputRow, parsers)]

def parseRowsWith(reader, parsers):
    for row in reader:
        yield parseRow(row, parsers)

def tryIfNone(f):
    def fOrNone(x):
        try:
            return f(x)
        except: return None
    return fOrNone

def tryParseField(fieldName, value, parserDict):
    parser = parserDict.get(fieldName)
    if parser is not None:
        return tryIfNone(parser)(value)
    else:
        return value

def parseDict(inputDict, parserDict):
    return {fieldName : tryParseField(fieldName, value, parserDict)
            for fieldName, value in inputDict.items()}

# Grouper Function
def picker(fieldName):
    return lambda row: row[fieldName]

# Grouper Function
def pluck(fieldName, rows):
    return map(picker(fieldName), rows)

def groupBy(grouper, rows, valueTransformation = None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[grouper(row)].append(row)

    if valueTransformation is None:
        return grouped
    else:
        return {key : valueTransformation(rows)
                for key, rows in grouped.items()}

# Example
"""
max_price_by_symbol = groupBy(picker("symbol", data, lambda rows: max(pluck("closing_price"), rows)))
"""