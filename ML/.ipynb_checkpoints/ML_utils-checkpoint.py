"""Utility functions."""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as skmetrics
import umap
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold,LeaveOneOut
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import RandomOverSampler



#import plotly.express as px
#import plotly.graph_objs as go
from IPython.display import display
from ipywidgets import VBox



def scatter_sel(
    df: pd.DataFrame,
    hover_label: str = None,
    hover_more_labels: list = None,
    color_label: str = None,
    marker_label: str = None,
    point_size: str = None,
    text_label: str = None,
    opacity: float = 1,
    x: str = "x",
    y: str = "y",
    z: str = "z",
    trend: str = None,
    x_error_bars: str = None,
    y_error_bars: str = None,
    x_range: list = None,
    y_range: list = None,
    z_range: list = None,
    width: int = None,
    height: int = None,
    animation: list = [None, None],
    cube_aspect_mode: bool = False,
    n_dims: int = 2,
) :




    """Plotly scatter plot wrapper.
 
    :param df: Input pandas dataframe.
    :type df: pd.DataFrame


    :param hover_label: Column name whose data is to be shown when hovering over a given data point.
    :type hover_label: str


    :param color_label: Column name whose data is to be used for coloring the data points, defaults to None.
    :type color_label: str, optional


    :param marker_label: Column name whose data is to be used for changing the marker of the data points, defaults to None.
    :type marker_label: str, optional

    

    :param point_size: Column name whose data is used for point sizes, defaults to None
    :type point_size: str, optional

    
    :param text_label: Column name whose data is used for labelling points in the plot, defaults to None
    :type text_label: str, optional


    :param trend: Add trendlines for each color / symbol combination, should be either 'ols' or 'lowess'. Only works for 2D plots.
    :type text_label: str, optional


    :param opacity: Float between 0 and 1. Sets opacity of points. Defaults to 1.

    :type n_dims: float, optional

    

    :param x: First coordinate, defaults to "x".

    :type x: str, optional

    

    :param y: Second coordinate, defaults to "y".

    :type y: str, optional

    

    :param z: Third coordinate, defaults to "z".

    :type z: str, optional

    

    :param x_error_bars: Column name whose data is used adding symmetrical x-axis error bars to each point, only works with 2d plots, defaults to None

    :type x_error_bars: str, optional

    

    :param y_error_bars: Column name whose data is used adding symmetrical y-axis error bars to each point, only works with 2d plots, defaults to None

    :type y_error_bars: str, optional

    

    :param x_range: List [float,float] of min and max value for the x-axis. This removes auto-scaling. Defaults to None.

    :type x_range: list, optional

    

    :param y_range: List [float,float] of min and max value for the y-axis. This removes auto-scaling. Defaults to None.

    :type y_range: list, optional

    

    :param z_range: List [float,float] of min and max value for the z-axis. This removes auto-scaling. Defaults to None.

    :type z_range: list, optional

    

    :param animation: List [str,str] of Column Names which correspond to Frame and Group. Frame is the levels of the slider for the animation e.g. 1,2,3,... Group provides object permanecy e.g. Sample1, Sample2, Sample3. Defaults to None.

    :type animation: tupe, optional

    

    :param cube_aspect_mode: Enables aspect mode as cube for 3d plots. Looks nice. You might want this for animated 3d plots. Defaults to False.

    :type cube_aspect_mode: bool, optional

    

    :param n_dims: Number of dimensions to plot, defaults to 2.

    :type n_dims: int, optional

    """



 



    assert n_dims in [2, 3], "'n_dims' should be either 2 or 3"

    

    assert x in df.columns, f"'x' is not a column in data frame, currently {x}"

    assert y in df.columns, f"'y' is not a column in data frame, currently {y}"

    if n_dims == 3: assert(z in list(df.columns)), f"'z' must be a column in data frame when n_dims=3, currently {z}"    

    assert 0 <= opacity <= 1, f"'opacity' must be between 0 and 1, currently {opacity}"

    #if cube_aspect_mode: assert n_dims == 3, "'cube_aspect_mode' is only available for 3D plots"

    if any(x is not None for x in (x_error_bars, y_error_bars)): assert n_dims == 3, "'error_bars' not available for n_dims=3"



 



    if point_size: assert df[point_size].max() != df[point_size].min(), "'point_size must' not be same value"

    if trend: assert(n_dims==2), "'trend_line' only available for 2d"



 



    if n_dims == 2:

        fig = go.FigureWidget(px.scatter(

            df,

            x=x,

            y=y,

            hover_name=hover_label,

            hover_data=hover_more_labels,

            color=color_label,

            symbol=marker_label,

            size=point_size,

            text=text_label,

            trendline=trend,

            opacity=opacity,

            error_x=x_error_bars,

            error_y=y_error_bars,

            range_x=x_range,

            range_y=y_range,

            animation_frame=animation[0],

            animation_group=animation[1],

            color_continuous_scale=px.colors.sequential.Cividis_r,

            width=width,

            height=height,

            

        ))

        

        tbl = go.FigureWidget(

                [

                    go.Table(

                        header=dict(values=hover_more_labels),

                        cells=dict(

                            values=[df[col] for col in hover_more_labels]

                        ),

                    )

                ]

            )



        def selection_fn(trace, points, selector):

            display('cio')

            display(trace)

            display(points)

            tbl.data[0].cells.values = [

                ['CC' 'UNK' 'UNK' 'UNK' 'UNK' 'UNK' 'UNK']

                ['UNK' 'UNK' 'UNK' 'UNK' 'UNK' 'UNK' 'UNK']

                ['UNK' 'UNK' 'UNK' 'UNK' 'UNK' 'UNK' 'UNK']

                ['UNK' 'UNK' 'UNK' 'UNK' 'UNK' 'UNK' 'UNK']

                #    df.loc[points.point_inds][col] for col in hover_more_labels

                ]



        scatter_sel = fig.data[0]

        

        scatter_sel.on_selection(selection_fn)





        

        #fig.update_layout(hover_data="nnc")

        

        if cube_aspect_mode:

            fig.update_yaxes(

                scaleanchor = "x",

                scaleratio = 1,

            )



 



    else:

        fig = px.scatter_3d(

            df,

            x=x,

            y=y,

            z=z,

            hover_name=hover_label,

            hover_data=hover_more_labels,

            color=color_label,

            symbol=marker_label,

            size=point_size,

            text=text_label,

            opacity=opacity,

            range_x=x_range,

            range_y=y_range,

            range_z=z_range,

            animation_frame=animation[0],

            animation_group=animation[1],

            color_continuous_scale=px.colors.sequential.Cividis_r,

        )

        

        tbl=None



 



        if cube_aspect_mode:

            fig.update_layout(scene_aspectmode="cube")



 



    #fig.show()

    return (fig, tbl)

 



def scatter(

    df: pd.DataFrame,

    hover_label: str = None,

    hover_more_labels: list = None,

    color_label: str = None,

    marker_label: str = None,

    point_size: str = None,

    text_label: str = None,

    opacity: float = 1,

    x: str = "x",

    y: str = "y",

    z: str = "z",

    trend: str = None,

    x_error_bars: str = None,

    y_error_bars: str = None,

    x_range: list = None,

    y_range: list = None,

    z_range: list = None,

    width: int = None,

    height: int = None,

    animation: list = [None, None],

    cube_aspect_mode: bool = False,

    n_dims: int = 2,

) -> None:



 



    """Plotly scatter plot wrapper.



 



    :param df: Input pandas dataframe.

    :type df: pd.DataFrame

    

    :param hover_label: Column name whose data is to be shown when hovering over a given data point.

    :type hover_label: str

    

    :param color_label: Column name whose data is to be used for coloring the data points, defaults to None.

    :type color_label: str, optional

    

    :param marker_label: Column name whose data is to be used for changing the marker of the data points, defaults to None.

    :type marker_label: str, optional

    

    :param point_size: Column name whose data is used for point sizes, defaults to None

    :type point_size: str, optional

    

    :param text_label: Column name whose data is used for labelling points in the plot, defaults to None

    :type text_label: str, optional

    

    :param trend: Add trendlines for each color / symbol combination, should be either 'ols' or 'lowess'. Only works for 2D plots.

    :type text_label: str, optional

    

    :param opacity: Float between 0 and 1. Sets opacity of points. Defaults to 1.

    :type n_dims: float, optional

    

    :param x: First coordinate, defaults to "x".

    :type x: str, optional

    

    :param y: Second coordinate, defaults to "y".

    :type y: str, optional

    

    :param z: Third coordinate, defaults to "z".

    :type z: str, optional

    

    :param x_error_bars: Column name whose data is used adding symmetrical x-axis error bars to each point, only works with 2d plots, defaults to None

    :type x_error_bars: str, optional

    

    :param y_error_bars: Column name whose data is used adding symmetrical y-axis error bars to each point, only works with 2d plots, defaults to None

    :type y_error_bars: str, optional

    

    :param x_range: List [float,float] of min and max value for the x-axis. This removes auto-scaling. Defaults to None.

    :type x_range: list, optional

    

    :param y_range: List [float,float] of min and max value for the y-axis. This removes auto-scaling. Defaults to None.

    :type y_range: list, optional

    

    :param z_range: List [float,float] of min and max value for the z-axis. This removes auto-scaling. Defaults to None.

    :type z_range: list, optional

    

    :param animation: List [str,str] of Column Names which correspond to Frame and Group. Frame is the levels of the slider for the animation e.g. 1,2,3,... Group provides object permanecy e.g. Sample1, Sample2, Sample3. Defaults to None.

    :type animation: tupe, optional

    

    :param cube_aspect_mode: Enables aspect mode as cube for 3d plots. Looks nice. You might want this for animated 3d plots. Defaults to False.

    :type cube_aspect_mode: bool, optional

    

    :param n_dims: Number of dimensions to plot, defaults to 2.

    :type n_dims: int, optional

    """



 



    assert n_dims in [2, 3], "'n_dims' should be either 2 or 3"

    

    assert x in df.columns, f"'x' is not a column in data frame, currently {x}"

    assert y in df.columns, f"'y' is not a column in data frame, currently {y}"

    if n_dims == 3: assert(z in list(df.columns)), f"'z' must be a column in data frame when n_dims=3, currently {z}"    

    assert 0 <= opacity <= 1, f"'opacity' must be between 0 and 1, currently {opacity}"

    #if cube_aspect_mode: assert n_dims == 3, "'cube_aspect_mode' is only available for 3D plots"

    if any(x is not None for x in (x_error_bars, y_error_bars)): assert n_dims == 3, "'error_bars' not available for n_dims=3"



 



    if point_size: assert df[point_size].max() != df[point_size].min(), "'point_size must' not be same value"

    if trend: assert(n_dims==2), "'trend_line' only available for 2d"





    if n_dims == 2:

        fig = go.FigureWidget(px.scatter(

            df,

            x=x,

            y=y,

            hover_name=hover_label,

            hover_data=hover_more_labels,

            color=color_label,

            symbol=marker_label,

            size=point_size,

            text=text_label,

            trendline=trend,

            opacity=opacity,

            error_x=x_error_bars,

            error_y=y_error_bars,

            range_x=x_range,

            range_y=y_range,

            animation_frame=animation[0],

            animation_group=animation[1],

            color_continuous_scale=px.colors.sequential.Cividis_r,

            width=width,

            height=height,

            

        ))

        

        #fig.update_layout(hover_data="nnc")

        

        if cube_aspect_mode:

            fig.update_yaxes(

                scaleanchor = "x",

                scaleratio = 1,

            )



 



    else:

        fig = px.scatter_3d(

            df,

            x=x,

            y=y,

            z=z,

            hover_name=hover_label,

            hover_data=hover_more_labels,

            color=color_label,

            symbol=marker_label,

            size=point_size,

            text=text_label,

            opacity=opacity,

            range_x=x_range,

            range_y=y_range,

            range_z=z_range,

            animation_frame=animation[0],

            animation_group=animation[1],

            color_continuous_scale=px.colors.sequential.Cividis_r,

        )



 



        if cube_aspect_mode:

            fig.update_layout(scene_aspectmode="cube")



 



    #fig.show()

    

    return fig

 



def fetch_scores(search, n_folds):

    return pd.DataFrame(

        [search.cv_results_[f"split{x}_test_score"] for x in range(n_folds)],

        index=[f"split{x}_test_score" for x in range(n_folds)],

    ).T.iloc[search.best_index_]





def print_scores(scores):

    print(

        scores,

        f"\nAccuracy mean: {scores.mean():.3f}",

        f"\nAccuracy std: {scores.std():.3f}",

    )





def classification2(emb, df_train_test, df_mild, models, n_folds=10,average_setting='binary', shuffle=True, random_state=42, verbose=True):



    kf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)



    d = {

        "model": [],

        "fold": [],

        "accuracy": [],

        "sensitivity": [],

        "specificity": [],

    }

    

    d2 = {

        "model": [],

        "fold": [],

        "accuracy": [],

        'X_test': [],

        'y_test': [],

        'y_pred': [],

        'prob class 0': [],

        'prob class 1': [],

        

    }





    # Data:

    X=df_train_test

    y=df_train_test

    

    X_mild = df_mild

    y_mild = df_mild

    



    fold = 0    

    for train, test in kf.split(X, y):



        if verbose:

            print(f"Fold #{fold}")



        X_train, X_test0, y_train, y_test = X[train], X[test], y[train], y[test]

        

        X_train_mild, X_test_mild, y_train_mild, y_test_mild = X_mild[train], X_mild[test], y_mild[train], y_mild[test]

        



        y_train = X_train

        X_train = X_train[emb].values

        X_train = np.array([list(i) for i in X_train])

        

        y_test = X_test0

        X_test = X_test0[emb].values

        X_test = np.array([list(i) for i in X_test])

        

        

        # --- For making of plot with mildly-polyreactive Abs,

        X_test = np.concatenate([X_test,X_mild])

        y_test = np.concatenate([y_test,y_mild])

        

        # --- For making of plot with mildly-polyreactive Abs,

        X_test_comb = pd.concat([X_test0,X_test_mild])

    

        y_test = X_test_comb

        X_test = X_test_comb[emb].values

        X_test = np.array([list(i) for i in X_test])



        for model_name, model in models.items():



            if verbose:

                print(f"Processing {model_name}")



            model = clone(model)  # reset weights while keeping input params

            

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            y_pred_prob_0 = model.predict_proba(X_test)[:,0]

            y_pred_prob_1 = model.predict_proba(X_test)[:,1]



            d["model"].append(model_name)

            d["fold"].append(fold)

            d["accuracy"].append(skmetrics.accuracy_score(y_test, y_pred))

            d["sensitivity"].append(skmetrics.recall_score(y_test, y_pred, pos_label=1, average=average_setting))

            d["specificity"].append(skmetrics.recall_score(y_test, y_pred, pos_label=0, average=average_setting))



            d2["model"].append(model_name)

            d2["fold"].append(fold)

            d2["accuracy"].append(skmetrics.accuracy_score(y_test, y_pred))

            d2['X_test'].append(X_test)

            d2['y_pred'].append(y_pred)

            d2['prob class 0'].append(y_pred_prob_0)

            d2['prob class 1'].append(y_pred_prob_1)

            d2["react"].append(y_test['react'])

            



        fold += 1



    return d, d2



def classification(X, y, models, n_folds=10,average_setting='binary', shuffle=True, random_state=42, verbose=True):



    kf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)



    d = {

        "model": [],

        "fold": [],

        "accuracy": [],

        "sensitivity": [],

        "specificity": [],

    }

    

    fold = 0

    for train, test in kf.split(X, y):



        if verbose:

            print(f"Fold #{fold}")



        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        

        #US = RandomUnderSampler()

        #US = RandomUnderSampler(sampling_strategy='majority', replacement=False,random_state=random_state)

        #US = RandomOverSampler()

        #X_train, y_train = US.fit_resample(X_train, y_train)

        

        

        for model_name, model in models.items():



            if verbose:

                print(f"Processing {model_name}")



            model = clone(model)  # reset weights while keeping input params

            

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            y_pred_prob_0 = model.predict_proba(X_test)[:,0]

            y_pred_prob_1 = model.predict_proba(X_test)[:,1]



            d["model"].append(model_name)

            d["fold"].append(fold)

            d["accuracy"].append(skmetrics.accuracy_score(y_test, y_pred))

            d["sensitivity"].append(skmetrics.recall_score(y_test, y_pred, pos_label=1, average=average_setting))

            d["specificity"].append(skmetrics.recall_score(y_test, y_pred, pos_label=0, average=average_setting))



          

        fold += 1



    return d



def loo_classification(X, y, models, n_folds=10, shuffle=True, random_state=42, verbose=True):



    kf = LeaveOneOut()



    d = {

        "model": [],

        "fold": [],

        "accuracy": [],

        #"sensitivity": [],

        #"specificity": [],

    }



    fold = 0

    for train, test in kf.split(X, y):



        if verbose:

            print(f"Fold #{fold}")



        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        

        #US = RandomUnderSampler()

        #US = RandomUnderSampler(sampling_strategy='majority', replacement=False,random_state=random_state)

        #US = RandomOverSampler()

        #X_train, y_train = US.fit_resample(X_train, y_train)



        for model_name, model in models.items():



            if verbose:

                print(f"Processing {model_name}")



            model = clone(model)  # reset weights while keeping input params

            

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)



            d["model"].append(model_name)

            d["fold"].append(fold)

            d["accuracy"].append(skmetrics.accuracy_score(y_test, y_pred))

            #d["sensitivity"].append(skmetrics.recall_score(y_test, y_pred, pos_label=1))

            #d["specificity"].append(skmetrics.recall_score(y_test, y_pred, pos_label=0))

            

            print(skmetrics.accuracy_score(y_test, y_pred))



        fold += 1



    return d





def regression(X, y, models, n_folds=10, shuffle=True, random_state=42, verbose=True):



    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)



    d = {

        "model": [],

        "fold": [],

        "MSE": [],

        "MAE": [],

        "r2_score": [],

    }



    fold = 0

    for train, test in kf.split(X):



        if verbose:

            print(f"Fold #{fold}")



        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]



        for model_name, model in models.items():



            if verbose:

                print(f"Processing {model_name}")



            model = clone(model)  # reset weights while keeping input params

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)



            d["model"].append(model_name)

            d["fold"].append(fold)

            d["MSE"].append(skmetrics.mean_squared_error(y_test, y_pred))

            d["MAE"].append(skmetrics.mean_absolute_error(y_test, y_pred))

            d["r2_score"].append(skmetrics.r2_score(y_test, y_pred))



        fold += 1



    return d





def draw_umap(

    data,

    color=None,

    target=None,

    test=None,

    n_neighbors=15,

    min_dist=0.1,

    n_components=2,

    metric="euclidean",

    title="",

):

    fit = umap.UMAP(

        n_neighbors=n_neighbors,

        min_dist=min_dist,

        n_components=n_components,

        metric=metric,

        random_state=42,

    )



    trans = None



    u = fit.fit_transform(data, y=target)



    if test is not None:

        trans = fit.transform(test)



    fig = plt.figure()



    if n_components == 1:

        ax = fig.add_subplot(111)

        ax.scatter(u[:, 0], range(len(u)), c=color)

        if test is not None:

            ax.scatter(trans[:, 0], range(len(u)), c="grey", s=200)



    elif n_components == 2:

        ax = fig.add_subplot(111)

        ax.scatter(u[:, 0], u[:, 1], c=color)

        if test is not None:

            ax.scatter(trans[:, 0], trans[:, 1], c="grey", s=200)



    elif n_components == 3:

        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=color, s=100)

        if test is not None:

            ax.scatter(trans[:, 0], trans[:, 1], trans[:, 2], c="grey", s=200)



    plt.title(title, fontsize=18)

    plt.show()



    return fit





def calc_stats(df, cols, group_by):



    # do mean

    df_mean = df.groupby(by=group_by, as_index=False).mean()



    # do std

    df_std = df.groupby(by=group_by, as_index=True).std().reset_index(drop=True)



    # remove fold column & fix column names

    df_mean = df_mean[[col for col in df_mean.columns if col != "fold"]]

    df_mean.rename(columns={col: f"{col}_mean" for col in cols}, inplace=True)



    # remove fold column & fix column names

    df_std = df_std[[col for col in df_std.columns if col != "fold"]]

    df_std.rename(columns={col: f"{col}_std" for col in cols}, inplace=True)



    return pd.concat([df_mean, df_std], axis=1)





def barplot(df, title=None, task="classification"):



    task = task.lower()

    allowed_tasks = ["classification", "regression"]

    assert task in allowed_tasks, f"'{task}' is not valid, must be one of '{allowed_tasks}'"



    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 7.5), constrained_layout=True)



    if title is not None:

        fig.suptitle(title, fontsize=14)



    for row in range(2):

        for col in range(2):

            if task == "classification":

                metric = [["accuracy", "sensitivity"], ["specificity", "specificity"]][row][col]

            else:

                metric = [["MSE", "MAE"], ["r2_score", "r2_score"]][row][col]

            g = sns.barplot(x="feat", y=metric, hue="model", data=df, ax=axes[row, col])



            for label in g.get_xticklabels():

                label.set_rotation(45)



            axes[row, col].get_legend().remove()

            axes[row, col].set_ylim(0, 1)



    plt.delaxes(axes[1, 1])



    axes[0, 1].legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.0)

    plt.show()
