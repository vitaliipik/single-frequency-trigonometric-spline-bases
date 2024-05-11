import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL._imaging import display
from mpl_interactions import zoom_factory, panhandler
from mpl_interactions import interactive_plot


def plot_basis(control_points, curve_points, is_control=True,text='tb-spline Curve'):
    if len(curve_points[0]) == 3:

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the curve
        if is_control:
            ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], 'ro-',
                    label='Three-dimensional three-leaf rose curve')
        ax.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2], 'b-',
                label='Three-dimensional three-leaf rose curve')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Three-dimensional three-leaf rose curve')

        # Add a legend
        ax.legend()


    else:

        curve_points = np.array(curve_points)
        if is_control:
            plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points')

        plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label=text)
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(text)
        plt.grid(True)
        plt.axis('equal')
    plt.show()


def line_plot(table):
    with plt.ioff():
        figure, axis = plt.subplots()
    for i in range(1,len(table.iloc[0])):
        interactive_plot(  table.iloc[:, 0],table.iloc[:, i], 'o-',label=table.columns[i])
    plt.legend()
    # plt.xlabel('Time Elapsed (seconds)')
    # plt.ylabel('number of points')
    plt.xlabel('number of points')
    plt.ylabel('Time Elapsed (seconds)')
    plt.title('time for different method')
    plt.legend()

    plt.tight_layout()
    disconnect_zoom = zoom_factory(axis)
    pan_handler = panhandler(figure)

    # plt.grid(True)
    # plt.axis('equal')
    plt.show()
from lets_plot import *



def line_plot_upd(table):

    df = pd.melt(table, id_vars=['number'], value_vars=table.columns[1:])
    p = (ggplot(df,aes(x="number",y="value", group='variable'))+ \
    geom_line(aes(color='variable'), size=1, alpha=0.5)+
         geom_point()+ scale_fill_brewer(type='seq'))

    p.show()
    # plt.grid(True)
    # plt.axis('equal')
def bar_plot(table,number,text="speedup",y_tick=False):
    bar_width = 0.15
    index = number
    index = np.arange(len(index))
    for i in range(len(table.iloc[0])):
        plt.bar(index + i * bar_width, table.iloc[:,i], bar_width,label=table.columns[i])
    plt.title('Performance Comparison')
    plt.xlabel('Dataset')
    plt.xticks(index + bar_width * (len(table) - 1) / 2, number)
    if y_tick:
        plt.yticks(np.arange(0, 3800, 300))
    plt.ylabel(text)
    plt.legend(title='Method')
    plt.tight_layout()
    plt.show()


def bar_plot_upt(table, number, text="speedup"):
    bar_width = 0.15
    index = table.index
    index = np.arange(len(index))

    p=    ggplot(table) + \
        geom_bar(aes(x='number', y='y', fill='Method'), stat='identity', position='dodge', width=bar_width) + \
        scale_fill_discrete(name='Method') + \
        ggtitle('Performance Comparison') + \
        xlab('Dataset') + \
        ylab(text)
    p.show()