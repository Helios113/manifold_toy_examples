ResBlock0 in,
ResBlock1 in,
ResBlock1 out,


Generate two sets of degenerate weights (DW)
a. One where one point fails
b. One where we have a fold in space
Save the weights as well so we don't have to regenerate


Plot Series 1:

3 PCA on ResBlock0 in
Plot structure in all resblocks
One plot per training step

Plot Series 2:

3 PCA on ResBlock0 in
Plot structure in all resblocks
One plot for DW.a and DW.b each

Plot Series 3:

Plot the cos sim of the kernel vector of Jf (vector of smallest singular value) with TpM
for ResBlock0 in and ResBlock1 in
Use the original unembedded structure for the plot
Plot a histogram of the cos sim, including min and max under each of the spheres
Plot a scatter plot underneath the histogram of the cos sim vs smallest singular value at that point

Plot Series 4:

Plot the smallest singular value of Jf not on a log scale
for ResBlock0 in and ResBlock1 in
Use the original unembedded structure for the plot

Compute lower lipshitz values, so for the every point, get the full distance matrix to all other points. Compute the distance in the input space and the output space. the ratio of output to input is lower lipshitz values, 

plot a scatter plot under the visualisation of input (on x axis) and output (on y axis) of the distances and then plot a histogram underneath the scatter of the kl values

Plot Series 4A:
The same as plot series 4 but for DW.a and DW.b

Plot Series 5:

Plot the UMAP strucutre
for ResBlock0 in, ResBlock1 in, ResBlock1 out
Use the original unembedded structure for the plot





for all series make a plotly interactive plot, save a view as png for all and also save the interactive html.