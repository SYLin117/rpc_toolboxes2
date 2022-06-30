import pandas as pd

if __name__ == "__main__":
    df = pd.read_excel('../tmp.xlsx')
    # print(df.head(3))

    df['improve'] = df.syn3 - df.actual
    # print(df.head(3))
    # print(df[df.actual > .9].improve.mean())  # 0.0063
    # print(df[df.actual < .9].improve.mean())  # 0.0327
    # print(df[df.actual < .8].improve.mean())  # 0.0430
    # print(df[df.actual < .7].improve.mean())  # 0.0534

    print(df.sort_values(by=['improve'], ascending=False).head(10))
    """
    199        200_stationery   0.675  0.698  0.698  0.747    0.072
    140               141_gum   0.735  0.778  0.777  0.793    0.058
    165  166_personal_hygiene   0.843  0.856  0.869  0.899    0.056
    144             145_candy   0.798  0.800  0.810  0.847    0.049
    21          22_dried_food   0.889  0.911  0.921  0.937    0.048
    149             150_candy   0.830  0.848  0.868  0.870    0.040
    128         129_chocolate   0.824  0.832  0.839  0.862    0.038
    198        199_stationery   0.877  0.890  0.890  0.913    0.036
    123         124_chocolate   0.889  0.901  0.912  0.925    0.036
    197        198_stationery   0.620  0.628  0.631  0.655    0.035
    """
