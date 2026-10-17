import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")
URL="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
raw=pd.read_csv(URL); df=raw.copy()
R={}
R["shape"]=raw.shape
R["missing"]=raw.isnull().sum().to_dict()
R["missing_pct"]=(raw.isnull().mean()*100).round(1).to_dict()
R["dups"]=int(raw.duplicated().sum())
R["dtypes"]={k:str(v) for k,v in raw.dtypes.items()}
R["describe"]=raw.describe().round(2).to_dict()

# Fig1 missing heatmap
plt.figure(figsize=(8,5)); sns.heatmap(raw.isnull(),cbar=False,yticklabels=False,cmap="viridis")
plt.title("Missing Values Before Cleaning (yellow = missing)"); plt.tight_layout(); plt.savefig("fig1_missing.png",dpi=150); plt.close()

# Cleaning
df=df.drop_duplicates()
df["HasCabin"]=df["Cabin"].notna().astype(int)
df=df.drop(columns=["Cabin"])
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Title"]=df["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
df["Title"]=df["Title"].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})
rare=df["Title"].value_counts()[lambda s:s<10].index
df["Title"]=df["Title"].replace(dict.fromkeys(rare,"Rare"))
R["age_median_overall"]=float(df["Age"].median()); R["age_mean_overall"]=float(df["Age"].mean()); R["age_skew"]=float(df["Age"].skew())
R["age_median_by_title"]=df.groupby("Title")["Age"].median().to_dict()
df["Age"]=df["Age"].fillna(df.groupby(["Title","Pclass"])["Age"].transform("median"))
df["Age"]=df["Age"].fillna(df["Age"].median())
for c in ["Survived","Pclass"]: df[c]=df[c].astype("category")
for c in ["Sex","Embarked","Title"]: df[c]=df[c].astype("category")
R["missing_after"]=int(df.isnull().sum().sum())
# outliers
def iqr(s):
    q1,q3=s.quantile([.25,.75]); i=q3-q1; return int(((s<q1-1.5*i)|(s>q3+1.5*i)).sum()), float(q3+1.5*i)
R["fare_outliers"]=iqr(df["Fare"]); R["age_outliers"]=iqr(df["Age"])
R["fare_skew"]=float(df["Fare"].skew())
df["FamilySize"]=df["SibSp"]+df["Parch"]+1
df["FareLog"]=np.log1p(df["Fare"])
df["AgeGroup"]=pd.cut(df["Age"],[0,12,18,35,60,100],labels=["Child","Teen","Young Adult","Adult","Senior"])
df.to_csv("titanic_clean.csv",index=False)

# Fig2 distributions
fig,ax=plt.subplots(1,3,figsize=(15,4.5))
sns.histplot(df["Age"],kde=True,bins=30,ax=ax[0],color="#3b6ea5"); ax[0].set_title("Age Distribution (after imputation)")
sns.histplot(df["Fare"],kde=True,bins=40,ax=ax[1],color="#c4694b"); ax[1].set_title("Fare Distribution (right-skewed)")
sns.boxplot(data=df,x="Pclass",y="Fare",ax=ax[2],palette="Set2"); ax[2].set_yscale("log"); ax[2].set_title("Fare by Class (log scale)")
plt.tight_layout(); plt.savefig("fig2_distributions.png",dpi=150); plt.close()

# Fig3 correlation
num=df[["Survived","Pclass","Age","SibSp","Parch","Fare","FamilySize","HasCabin"]].copy()
num["Survived"]=num["Survived"].astype(int); num["Pclass"]=num["Pclass"].astype(int)
corr=num.corr()
plt.figure(figsize=(8,6.5)); sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0,square=True)
plt.title("Correlation Matrix"); plt.tight_layout(); plt.savefig("fig3_corr.png",dpi=150); plt.close()
R["corr_surv"]=corr["Survived"].round(3).to_dict()

# Fig4 survival by sex & class
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
sns.barplot(data=num.assign(Sex=df["Sex"]),x="Pclass",y="Survived",hue="Sex",ax=ax[0],errorbar=None,palette="Set1")
ax[0].set_title("Survival Rate by Class and Sex"); ax[0].set_ylabel("Survival rate")
s=df.groupby("AgeGroup",observed=True)["Survived"].apply(lambda x:(x.astype(int)).mean())
sns.barplot(x=s.index,y=s.values,ax=ax[1],palette="crest"); ax[1].set_title("Survival Rate by Age Group"); ax[1].set_ylabel("Survival rate")
plt.tight_layout(); plt.savefig("fig4_survival.png",dpi=150); plt.close()
R["surv_overall"]=float(num["Survived"].mean())
R["surv_sex"]=df.groupby("Sex",observed=True)["Survived"].apply(lambda x:x.astype(int).mean()).round(3).to_dict()
R["surv_class"]=df.groupby("Pclass",observed=True)["Survived"].apply(lambda x:x.astype(int).mean()).round(3).to_dict()
R["surv_age"]=s.round(3).to_dict()
R["surv_cabin"]=df.groupby("HasCabin")["Survived"].apply(lambda x:x.astype(int).mean()).round(3).to_dict()
R["surv_embarked"]=df.groupby("Embarked",observed=True)["Survived"].apply(lambda x:x.astype(int).mean()).round(3).to_dict()
R["surv_family"]=df.assign(fs=pd.cut(df["FamilySize"],[0,1,4,20],labels=["Alone","2-4","5+"])).groupby("fs",observed=True)["Survived"].apply(lambda x:x.astype(int).mean()).round(3).to_dict()
R["sex_counts"]=df["Sex"].value_counts().to_dict(); R["class_counts"]={int(k):int(v) for k,v in df["Pclass"].value_counts().items()}
R["embarked_counts"]=df["Embarked"].value_counts().to_dict()
R["desc_clean"]=df[["Age","Fare","SibSp","Parch","FamilySize"]].describe().round(2).to_dict()
R["fare_median_class"]=df.groupby("Pclass",observed=True)["Fare"].median().round(2).to_dict()
json.dump(R,open("results.json","w"),indent=1,default=str)
print(json.dumps(R,indent=1,default=str))
