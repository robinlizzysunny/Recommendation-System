import pandas as pd
from model import preProcessing
from flask import Flask, render_template, request

app = Flask(__name__)

df = pd.read_csv("./Data/sample30.csv")
fin_ratings = pd.read_csv("./Data/recom.csv")
fin_ratings = fin_ratings.set_index('userId')



@app.route("/", methods=['POST', "GET"])
def home():
    if request.method == 'POST':
        user = request.form['name']
        if user in fin_ratings.index.tolist():
            lis = fin_ratings.loc[user].sort_values(ascending=False).index[:20]
            df_recom = df[df['name'].isin(lis)]
            fin_df = preProcessing(df_recom)
            fin_df = fin_df[['name', 'preds']]
            d = fin_df.groupby('name').mean().sort_values(ascending=False, by="preds")*100
            # d = df.loc[user].sort_values(ascending=False)
            products = d[:5].index.tolist()
            return render_template('index.html', products=products, submit="yes")
        else:
            return render_template('index.html', products="None")
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)