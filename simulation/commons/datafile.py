import csv

# Simple CSV Wrapper Module
# Written by Ali Seyhun Saral www.saral.it
# Keeps things simple and consistent in columns when you need to dump data to a CSV file continuously

# IF YOU ARE PARALLELIZING, MAKE SURE TO USE A DIFFERENT FILENAME FOR EACH THREAD
# Otherwise, you files might get corrupted

class DataFile:
    def __init__(self, filename, columns=None):
        self.filename = filename
        self.writer = None

        if columns:
            self.set_columns(columns)
    
    def set_columns(self, col_list):
        self.columns = col_list
 
    def write_header(self):
        with open(self.filename, "w", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.columns)
            writer.writeheader()

    def write_line(self, **kwargs):
#        print(kwargs)
        with open(self.filename, "a", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.columns)
            writer.writerow(kwargs)

## Example
#datafile = DataFile(filename="newfiletest3.csv", columns = ["a", "b", "c"])

#datafile.write_header()

# datafile.write_line(a=3,c=1)
# datafile.write_line(c=2, a=2,b=1)

# Can also be supplied dict by **
#datafile.write_line(**dict(a=3,c=1))
#datafile.write_line(**dict(b=7,c=1))
#datafile.write_line(**dict(a=1,b=3,c=2))


