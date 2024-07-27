import math
import random
import numpy as np
import pandas as pd

random.seed(42)
# get name of file in cea csv file
def getNameCsvFile(path):
    """_summary_
    
    Args:
        path (file): _path of the cea target file_
    """
    df = pd.read_csv(path, header=None)   
    col1 = df[0]
    data = []
    cols_not = [] # contains no duplicate data
    # store each of the first column in a list
    for i in col1:
        data.append(i) 
    # remove duplicate key
    for i in data[0:]:
        if i not in cols_not:
            cols_not.append(i)
    
    return cols_not

def getAllCellInTableColByBCol(table, is_vertical, comma_in_cell):
    """_summary_
        
    Args:
        table (Dataframe): _The path of csv table_
        col_before_row (boolean): _Additional information for the specific semantic annotation task_
        comma_in_cell (boolean): _Specify if table have a lot of data for a given cell separate by(,)_
    """
    list_cell = []
    for cols in table.columns:
        for i, row in table.iterrows():
            if type(row[cols]) == type(np.nan):	
                list_cell.append(["NIL", "NIL"])
            else:
                """" 
                    take 10 elements randomly not nan in the row 
                """
                cols_row_not_nan = [x for x in row if not isinstance(x, float) or not math.isnan(x)]
                # check the comma in the cell
                if comma_in_cell:
                    cols_row_not_nan = [random.choice(str(x).split(",")) for x in cols_row_not_nan]
                
                if is_vertical:
                    # check if row a row contain more than 15 elements
                    if len(cols_row_not_nan) > 15:
                        choice_element = random.sample(cols_row_not_nan[1:], k=15)
                    else:
                        choice_element = cols_row_not_nan[1:]
                else:
                    if len(cols_row_not_nan) > 15:
                        choice_element = random.sample(cols_row_not_nan, k=15)
                    else:
                        choice_element = cols_row_not_nan
                list_cell.append([row[cols], choice_element])
    return list_cell

def getAllCellInTableRowByRow(table, comma_in_cell):
    """_summary_
        
    Args:
        table (Dataframe): _The path of csv table_
        col_before_row (boolean): _Additional information for the specific semantic annotation task_
        comma_in_cell (boolean): _Specify if table have a lot of data for a given cell separate by(,)_
    """
    list_cell = []
    for i, row in table.iterrows():
        for cols in table.columns:
            # print(row[cols])
            if type(row[cols]) == type(np.nan):	

                list_cell.append(["NIL", "NIL"])
            else:
                """ 
                    take the first 10 elements not nan in the row 
                """
                cols_row_not_nan = [x for x in row if not isinstance(x, float) or not math.isnan(x)]
                # if cell have more entity separated by coma take the first element
                if comma_in_cell == True:
                    cols_row_not_nan = [x.split(",") for x in cols_row_not_nan]
               
                choice_element = cols_row_not_nan[:10]
                list_cell.append([row[cols], choice_element])
    return list_cell


def tableToVectCol(table, writer, list_cell, filename, col_before_row):
    """_summary_

    Args:
        table (_Dataframe_): _The path of csv table_
        writer (_csvWriter_): _write cell table in other file as vector of cell_
        list_cell (_List_): _Cells of the table_
        filename (_string_): _Name of the file where table is located_
        col_before_row (_Boolean_): _Additional information for the specific semantic annotation task_
    """
    
    total_rows = len(table.axes[0])
    total_cols=len(table.axes[1])
    filetotalrowcol = total_rows * total_cols
    # print("File total size: ", filetotalrowcol)
    row = 0
    col = 0
    cell = 0
    while row < filetotalrowcol:
        if row < total_rows:
            if col_before_row == True:
                """_summary_           
                    Check the structure of your csv target file, if the col_id start at 0 the
                    the col variable will be col and not col+1 same for the row_id
                """
                writer.writerow([filename, col, row, list_cell[cell][0], list_cell[cell][1]])
                row += 1
                cell +=1
            else:
                writer.writerow([filename, row+1, col, list_cell[cell][0], list_cell[cell][1]])
                row += 1
                cell +=1
        else:
            row = 0
            filetotalrowcol -= total_rows
            col += 1

def tableToVectROw(table, writer, list_cell, filename, col_before_row):
    """_summary_

    Args:
        table (_Dataframe_): _The path of csv table_
        writer (_csvWriter_): _write cell table in other file as vector of cell_
        list_cell (_List_): _Cells of the table_
        filename (_string_): _Name of the file where table is located_
        col_before_row (_Boolean_): _Additional information for the specific semantic annotation task_
    """
    
    total_rows = len(table.axes[0])
    total_cols=len(table.axes[1])
    filetotalrowcol = total_rows * total_cols
    # print("File total size: ", filetotalrowcol)
    row = 0
    col = 0
    cell = 0 # index of each cell in list_cell
    while col < filetotalrowcol:
        #
        if col < total_cols: # allow to match any data in each column
            if col_before_row == True:
                """_summary_           
                    the cells are extracted in the table row by row. 
                """
                # print([filename, col+1, row, list_cell[cell][0], list_cell[cell][1]])
                writer.writerow([filename, row+1, col, list_cell[cell][0], list_cell[cell][1]])
                col += 1
                cell +=1
            # else:
            #     writer.writerow([filename, row+1, col, list_cell[cell][0], list_cell[cell][1]])
            #     row += 1
            #     cell +=1
        else:
            col = 0
            filetotalrowcol -= total_cols
            row += 1
