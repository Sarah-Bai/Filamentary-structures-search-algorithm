# Filamentary structures search algorithm


## Description

This project facilitates the search and extraction of maps that contain a single filamentary structure. The algorithm is based on Rolling Hough Transform [[1]](#1), which is a filamentary structures detection software, based on Hough transform [[2]](#2). The algorithm then detects the largest filament on the given map and returns a FITS file with the map of angles and the map of original data. The main advantage of the algorithm is the possibility to detect individual distinct filametns over large data sets. In case the goal is to detect multiple filamentary structures, appropriate changes can be applied at the labeling stage.

The algorithm was tested on the data, obtained from Planck Telescope at 353 GHz. The code was writted in Jupyter Notebook, a python-based IDE.

I ran the algorithm over the whole sky map above -60 and below 60 degrees. The algorithm was tested twice with different parameters and then visually analysed the output. Result:

|                     | length 7 width 27 | length 12 width 15 |
| :---:                |     :---:          |          :---:       |
| false detection     | 3352               | 3098                |
| ambiguous detection | 734                | 286                 |
| good detection      | 250                | 113                 |


For the Planck data I advice to use **length 7 and width 27** with sizes around 250x250 pixels.

---

### Schematic representation of the algorithm

![Scheme](https://github.com/Sarah-Bai/Filamentary-structures-search-algorithm/blob/main/scheme.png)

### Distribution of the *good filaments* on the whole sky map (length 7 and width 27)


![Map](https://github.com/Sarah-Bai/Filamentary-structures-search-algorithm/blob/main/all.png)

---


## Example

```
center_longitude = -60
center_latitude = -9
width = 250
length = 250
smr = 27
wlen = 7

folder = '/home/User/Desktop/output/'
data = 'HFI_SkyMap_353-psb_2048_R3.01_full_CIBsub_Nested_res7_I.fits'

main(center_longitude, center_latitude, width, length, smr, wlen, folder, data)
```

Please see the .ipynb file for a detailed example. Tables with coordinates of both good and abmbiguous detected filaments are added to the repository.
This work was performed in the frame of an undergraduate student research project at Nazarbayev University.



# References

<a id="1">[1]</a> 
S E Clark, J E G Peek, M E Putman. (2014). 
Magnetically Aligned HI Fibers and the Rolling Hough Transform 
M.E. 2014, ApJ, 789, 82


<b id="2">[2]</b> 
Hough, P. V. C. (1962).
Method and Means for Recognizing Complex Patterns
