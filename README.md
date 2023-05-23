### Information
Copyright (C) 2021, The Regents of the University of Michigan.

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contributers: Drew Vecchio, Samuel Mahler, Mark D. Hammig, Nicholas A. Kotov
Contact email: vecdrew@umich.edu
URL: https://github.com/drewvecchio/StructuralGT

### Structural Graph Theory Calculator

![](/Ref/SGT_BC_ex.png)

StructuralGT is designed as an easy-to-use python-based application for applying graph theory (GT) analysis to 
structural networks of a wide variety of material systems. This application converts digital images of 
nano-/micro-/macro-scale structures into a graph theoretical (GT) representation of the structure in the image 
consisting of nodes and the edges that connect them. Fibers (or fiber-like structures) are taken to represent edges, 
and the location where a fiber branches, or 2 or more fibers intersect are taken to represent nodes. The program 
operates with a graphical user interface (GUI) so that selecting images and processing the graphs are intuitive and 
accessible to anyone, regardless of programming experience.  Detection of networks from input images, the extraction of 
the graph object, and the subsequent GT analysis of the graph is handled entirely from the GUI, and a PDF file with the
results of the analysis is saved.

### Installation / Getting Started

StructuralGT is available as a downloadable python package from:
https://github.com/drewvecchio/StructuralGT

Or it can be pip installed from pypi.org using pip commands:
pip install StructuralGT.  This process should only take moments.

A standalone windows executable can be downloaded from:
https://drive.google.com/drive/folders/1p0NJEryrGDoZ7yvAXB5rfp08K5Muxeaa?usp=sharing \
The windows executable can be run by double-clicking the icon on supported Windows operating systems (this may take
up to a minute to load, so do not be alarmed).

Otherwise,
In the command line, open the StructuralGT GUI by running 'python StructuralGT'

### Selecting analysis mode

Upon starting up the StructuralGT GUI, a window with three buttons will appear.  It provides options to perform 
analysis on a single image or analysis on a batch of images, and a button to display copyright information.  Selecting
the single-image analysis option will generate the window for standard analysis with StructuralGT.  Using multi-image 
analysis provides the added benefit of performing calculation on a folder full of images in sequence, and compiling 
all results into a csv file.  However, using multi-image analysis does not allow for selecting settings individual to
each image, so the user should ensure that the image detection settings can be applied to all images in a file.

### Loading in Images

StructuralGT is designed to analyze greyscale images with a .tif, .png, or .jpg extension.
Electron microscopy images, such as those from SEM, TEM, and STEM are the intended focus for this program, but any 
image of these types that can be loaded as a grayscale image can be used.

For the single-image analysis, select an appropriate image by clicking the "Select file..." button, and using the 
dialog box that appears to search for the directory and file of interest. The default save location for the results 
file will be in the same directory as the input image, but this can be changed by clicking the 
"Choose Save Location..." button and finding the desired location in the file dialog that appears.
In many instance, a user will want to crop an image to select a specific region of interest, or to ignore a scale bar.
By clicking "Crop Image", a picture of the selected image will appear in the GUI, which then a region can be selected.
The crop region can be created by clicking and holding down the mouse button at the top left corner of the desired 
region, dragging the cursor to the desired bottom right corner, and releasing the mouse button. A red box will appear 
indicating the selected area. Alternatively, pixel coordinates can be used to manually adjust this region by entering 
the pairs of pixel coordinates into the boxes below the image, and clicking the "Update" button.
The images loaded into the GUI for cropping are scaled down to a max dimension of 512 pixels, if applicable.
Click "Proceed with crop" to accept the crop region as your image selection for the next step, or Click 
"Proceed without crop" to use the full picture for the analysis.
Clicking "Exit" will close the application.

With the multi-image version of the program, you will only be able to select a folder and target directory; 
no images can be cropped in-program for the multi-image analysis, and so all of the images in the desired folder will 
need to be appropriately cropped with an external program in advance.

### Running the tests

After selecting the image and clicking to proceed, the GUI will change to the StructuralGT Settings window.
The GUI is segmented into four quadrants, for choosing the image detection settings, for previewing the binary image, 
handling the generated graph object, and calculating the GT parameters using NetworkX algorithms.

![](/Ref/GUI_ex.png)

### Image Detection Settings

The primary feature of the image detection settings is three radio buttons for selecting the type of thresholding to 
apply to the end image.

**Global threshold** will isolate all pixels above and below the user-set threshold value, recommended for images with 
high contrast (STEM).\
**Adaptive threshold** determines locally how to threshold pixels, as opposed to using the same threshold for the global 
image.  Recommended for images where contrast between foreground and background is not distinct globally but can be 
distinguished locally (SEM).\
The **OTSU Threshold** uses Otsu's method of binarization to select a threshold that minimize variance between dark and 
light pixels.  Like an automated version of the global threshold, recommended for images that need a consistent 
threshold applied.

The global threshold value can be applied by a slider, from values of 1-255.
The adaptive threshold kernal size can be entered in the appropriate text box, as an odd integer from 3-511.

Several filters and computer vision techniques are available to preprocess the image for more accurate image detection.

The **gamma** of the image can be adjusted (from 0.01 to 5.00) by the slider at the top.  Adjusting gamma provides a 
nonlinear brightness adjustment.  Gamma values above 1 will make moderate pixel values brighter, while values below 1
will have the opposite effect.

A **gaussian blur** can be applied, and the size of the blurring element can be set by entering an odd integer into the 
corresponding text box.

A **median filter** can be applied, using a 3x3 kernel to reduce salt-and-pepper noise in the image.

The **low pass filter** option applies a Hamming Window filter to the image, using the window size entered in the text box.

**Sobel**, **Scharr**, and **Laplacian** gradients can be applied if selected.

**Autolevel** uses the blurring element kernel size to normalize the pixel values within that neighborhood. Can be helpful
in edge detection.

Clicking the _Foreground is dark_ button inverts the image.

There is a "Preview" button in the bottom-right of this section that will allow you to see a preview of the steps of 
this process, so that the proper settings can be used and adjusted before proceeding with the full analysis.
Clicking this button will open a window with four images to help pick the right settings.
The top-left image will be the unaltered source image.
The top-right image will be the processed image with any filters and blurs applied, but no thresholding.
The bottom-left contains a histogram of the pixel values in the image, which can be helpful for selecting a global 
threshold; a dashed line will appear showing the selected threshold if Global or OTSU threshold is selected.
The bottom-right image will be the final binary image.

Below the Image Detection Settings is also an updated preview of the binary image that will be obtained with the current
settings.  The goal is to produce an image in which the expected network appears in white and is connected. The 
background of the image should appear as black pixels.

For multi-image analysis, only the first image of the sequence will appear in the preview.  The settings selected here
will be applied to all images for the multi-image analysis.

### Graph Extraction Settings

After selecting the appropriate settings for obtaining the binary image, the next set of settings can be selected in 
order to tell the software how to handle the graph it extracts from the binary image.
The binary image will be converted to a skeleton, which is a single-pixel wide network that is the reduced form from 
the binary image.
The software identifies where the lines in this skeleton intersect and end as "nodes" for the graph, and the lines that
then connect these identified nodes as "edges".
However, this initial graph is often slightly erroneous in one of several ways that can be corrected for by using the 
settings in this section, as well as providing some additional options for handling the data.

Often in a graph with wide edges, an interesection that a human might identify to be a single node gets represented by 
a cluster of several intersections by the skeleton.
The option to _Merge nearby nodes_ applies a small disk element over each node, then attempts to reduce the skeleton 
again, with the intention of combining nodes that are within a few pixels of each other into the same single pixel 
(this method is imperfect, and has a limit to only several pixels, so it cannot work for nodes that are not very 
close to each other).

The _Prune dangling edges_ option looks to remove edges that do not connect to other nodes - this is useful especially 
for excluding edges that extend past the edge of the image, which reduces the informational content of that edge.

You can tell the software whether to include/exclude self-loops in the graph (a node with an edge connecting to itself),
or to allow the graph to be a multigraph (more than one edge connecting the same pair of nodes).  Be aware, some
calculations cannot be performed on multigraphs.  By definition, all graphs of structural networks are undirected.

Use _Remove disconnected segments_ in order to delete parts of the graph that are not connected to the main network 
either through noise or consequence of filtering.  Subgraph objects with a size less than the entered value are deleted.
It is highly recommended to remove all disconnected elements, so that only a singular connected subgraph remains.  
Be aware that some calculations cannot be performed if there is not only a single connected graph.

All graphs use an unweighted analysis (each edge treated equivalently with a value of 1), though a weighted analysis
can also be performed.  By selecting to _assign edge weights by diameter_, the width and length of each edge are stored, which can
be used to treat the edges non-uniformly.  The length is measured by a trace along the edge, and the width is
approximated by measuring the pixel width along the perpendicular bisector of the edge.

The graph data can be exported, if prompted, in two different forms.  The graph can be exported as an edge list, 
a csv file containing pairs of nodes that represent each edge in the graph.  If this is a weighted graph, the edge 
weight (width) and length are also reported for each edge.  It can also be exported as a Graph Exchange XML file, which
contains the graph information, which can be opened by graph analysis software, such as Gephi.

To help interpret these graph data objects better, the node IDs that are stored in these data objects can be printed
onto the figure in the PDF results, so that they can be visually connected.


### NetworkX Calculation Settings

Each checkbox will tell the software to compute the associated parameter using the NetworkX algorithm.
If the Graph Extraction Setting _Assign edge weights by diameters_ was checked, then some of these calculations will 
additionally calculate the equivalent weighted parameter.
Each calculation selected will be attempted on the final graph generated.
These parameters are briefly described below; more detailed information about these calculations can be found from the 
NetworkX documentation page.  https://networkx.org/documentation/stable/reference/algorithms/index.html

**Degree**: A count of how many edges connect to each node
Reported as: Average value, histogram, heat-map, and width-weighted counterparts

**Network diameter**: The number of edges traversed that you would never need to exceed to travel between any two nodes 
in the graph; also understood as the longest shortest path.
Reported as: value

**Graph density**: a measure of how many edges exist out of all possible edges that could in a complete graph.
Reported as: value

**Global efficiency**: The inverse of the distance to travel to each node.
Reported as: value

**Wiener Index**: The sum of all shortest path lengths in the graph
Reported as: value, and length-weighted counterpart

**Clustering coefficient**: Of a node, the fraction of neighbors of the node, that are connected directly to each 
other as well. Also understood as how many triangles form out of all possible tringles that could exist.
Reported as: average value, heat-map

**Nodal connectivity**: a measure of the minimum number of edges needed to be removed to disconnect a pair of nodes
Reported as: average value, maximum flow between periphery (width-weighted minimum cut)

**Assortativity coefficient**: a measure of similarity of connections within the graph with respect to the degree - this 
produces a value between -1 {not at all similar} and +1 {very similar}.  A value near zero represents randomly 
distributed connections.
Reported as: value, width-weighted counterpart

**Betweeness centrality**: a measure of how often a particular node lies along the shortest path between two other nodes 
Reported as: average value, histogram, heat-map, and width-weighted counterparts

**Closeness centrality**: the sum of the inverse shortest distance of a certain node to all other nodes in the graph
Reported as: average value, histogram, heat-map, and length-weighted counterparts

**Eigenvector centrality**: the solution to the eigenvector equation, Ax = \lambda x, such that all elements of x are 
positive (A is the adjacency matrix of the graph).  A measure of centrality where a node's centrality is dependent 
on the centrality of the nodes connected to it.
Reported as: average value, histogram, heat-map, and width-weighted counterparts

*_Graph Ricci Curvature_: The Ollivier-Ricci curvature and Forman-Ricci curvature can be evaluated using the python
package, GraphRicciCurvature.  The ricci curvature is a method of community segmentation through differential geometry
measures. Ricci curvature values are assigned to edges, with positive ricci curvature indicating connection to that 
node, and the community that node is a part of, and negative curvature indicating the nodes not being part of the same 
community.  Ollivier-Ricci curvature is based on optimal transport theory, and Forman-Ricci curvature is based topology.
Further information for ricci curvature can be found in the GraphRicciCurvature GitHub page.
https://github.com/saibalmars/GraphRicciCurvature
Note the installation requirements for GraphRicciCurvature on that page.  NetworKit >= 6.1 is required for shortest 
path algorithms, but is know to have trouble installing on some systems.  Help with NetworKit installation can be found 
here:
https://github.com/networkit/networkit#installation-instructions

Reported as: average value, heat-map, and length-weighted counterparts


_Exceptions_:
1) If more than one subgraph exists, network diameter and average nodal connectivity cannot be calculated, and NaN will 
   be reported in the results
2) If the graph is formatted as a multigraph, average nodal connectivity, betweenness centrality, and eigenvector 
   centrality cannot be evaluated, and their calculation will be skipped
3) If the Ricci curvature is selected to be measured, it cannot be measured if self-loop or multiple edges exist. This 
   will overwrite any selection made previously, so the graph will be extracted without self-loops or multiple edges.
   
### Results

The generated results for each image in either single-image or multi-image will be generated in a PDF with the same name
as the original file, followed by "_SGT_results.pdf".  It will be saved to the location previously designated.

The first page contains the same information that would appear when using "Preview" in the Image Detection settings.
This contains the original image, processed image (post image filtering), a histogram of pixel values of the processed 
image, and the binary image acquired after applying a thresholding method.

![](/Ref/SEM_graph_ex.png)

The second page contains the graph that was obtained from the binary image.
The top image shows the initial skeleton that was formed, with color coding to indentify what each pixel signified: 
Blue represents branch points, red represents end points, and white represents edges.
This can give the user a good idea of what happens when the graph extraction settings are applied, by seeing 
disconnected segments, and identifying dangling edges (an edge that connects to an endpoint).
The bottom image displays the final graph that is used for calculation, with all the graph extraction settings applied.
This graph is overlaid onto the original source image, so that the visual match is apparent, and is color coded with 
the nodes in blue and the edges in red.
If the user selected to display node IDs, then a small number displaying the node ID will be present next to the 
respective node in each image.
This is useful for correlating graph exported to Gephi to this image, as the node IDs should be consistent.

If the user did not select to perform weighted edge analysis, then the third page of the PDF results file will contain 
all the parameters selected for calculation (in addition to total # of nodes and edges) in a table, 
and any histograms would be displayed here as well.

If the user selected to perform weighted edge analysis, then the third page of the PDF results file will contain two 
tables: one table with the unweighted GT parameters, and a second table containing the weighted GT parameters.

Following the tables containing GT parameter values and the histograms of GT parameter data, heat-maps are displayed 
for the appropriately selected parameters (when applicable).  These heat-maps display the GT parameter for each node by
coloring each node in correspondence to its value. The color bar to the right displays how the node values correspond
to their color.
If ricci curvature maps were set to calculate, their heat-maps will follow.  These have their edges mapped to colors 
instead of the nodes, since orc and frc values are assigned to edges.

The final page of the PDF file will always be a summary of the analysis of StructuralGT.  It will contain the file name,
run date and time, and the settings used for image detection and graph extraction.  These settings can be helpful if 
an analysis needs to be repeated, so that the settings can be recalled.

### Example SGT Analysis

Included in the 'Example Images' Folder are test case images: an SEM image of an aramid fiber network, and a STEM image 
of network of self-assembled nanoparticles.  Included in this folder are additionally the result files produced 
according to the processing settings discussed below.  You may use these test images to verify proper operation of the 
StructuralGT package and to familiarize yourself with the GUI.

Open StructuralGT by any of the methods discussed in the 'Getting Started' portion of the Readme document.  After a 
short time, a window should pop-up requesting that you select either single-image or batch-image StructuralGT analysis. 
Click 'Single Image StructuralGT'.  This will bring up the main StructuralGT GUI.  From here, you can select the image 
file to process by clicking 'Select file...' and navigating to the location of the file that you wish to analyze 
(Example Images).  The result save directory will default to the same file path as the selected image file.  This can 
be changed using the 'Choose save location...' button.  If an image needs to be cropped, this can be done with the 
'Crop Image' button.  This is helpful for cropping an area that excludes the scale bar or other microscopy annotations. 
For the example images, cropping will not be needed.  Once the file path is properly selected, click 'Proceed without 
crop' to move onto the GUI containing the SGT processing settings.

**Test_STEM.png**\
This is a dark-field STEM image of a network of self-assembled nanoparticles.  The 2D nature of the network and high 
contrast between the sample and background makes it a convenient image to analyze.\
In 'Image Detection Settings':\
Adjust the gamma to 0.75 to reduce the brightness of the background.\
Apply a low-pass filter to further reduce the background noise and gradients in the brightness of the background.\
Keep the filter window size at its default value of 10 for this example.\
Ensure that a global threshold is being used, and adjust the global threshold value to 14.\
In 'Graph Extraction Settings':\
Assign weights by diameter.\
Select to removed disconnected segments. It is highly recommended to always use this option.\
The default value of 500 for remove object size is sufficient.\
Keep the 'Disable multigraph' box checked as it should be by default.  This prevents parallel edges, at least one of 
which is present in this graph.\
Select to display node IDs in the final graph.\
In 'NetworkX Calculation Settings':\
Press the 'Select All...' button, then click 'Proceed'.\
The calculation should only take a few seconds (~15 sec) to complete, as observed by the progress bar.\
Compare the result file generated with the suffix _SGT_results.pdf to the _example_results.pdf file.

**Test_SEM.tif**\
This is an SEM image of a network of nanofibers.  This is a 3D network with a lesser distinction between fibers in the 
foreground and the background.  The goal is to generate a graph of the 2D-projection of the top slice of the network.\
In 'Image Detection Settings':\
Adjust the gamma to 0.50 to reduce the brightness of the background.\
Apply median filter.\
Apply the gaussian filter.\
Keep the gaussian blur size at its default value of 3 for this example.\
This filters will smoothen the noise in the foreground of this image\
Select the adaptive threshold, which will better allow to distinguish foreground fibers from background fibers.\
Increase the size of the adaptive threshold kernel to 111 to compare across a wider area.\
In 'Graph Extraction Settings':\
Select 'Merge nearby nodes'.\
Select 'Prune dangling edges'.\
Select 'Remove self-loops'.\
Select to removed disconnected segments. It is highly recommended to always use this option.\
The default value of 500 for remove object size is sufficient.\
Uncheck the 'Disable multigraph' for this example to allow for parallel edges in the graph.\
In 'NetworkX Calculation Settings':\
Press the 'Select All...' button, then click 'Proceed'\
Due to the greater number of nodes, the calculation will take ~10 minutes to complete. as observed by the progress bar.\
Most of the computation time is due to calculating Average Nodal Connectivity, so you may optionally uncheck this box 
to make sure the results can be obtained in a few seconds.\
Compare the result file generated with the suffix _SGT_results.pdf to the _example_results.pdf file.
