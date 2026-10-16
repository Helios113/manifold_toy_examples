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

Plot the cos sim of the kernel vector with TpM
for ResBlock0 in and ResBlock1 in
Use the original unembedded structure for the plot


Plot Series 4:

Plot the UMAP strucutre
for ResBlock0 in, ResBlock1 in, ResBlock1 out
Use the original unembedded structure for the plot

