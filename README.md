# Computational Audio Analysis - Audio Feature Extraction

#### C2
![Plots](plots.png)


RMS mean v/s RMS std features provide the best seperation between the speech and music. We can clearly draw a boundary that mostly seperates the classes. This distinction can be attributed to the reason that speech has more pauses than music, which increase the RMS std value for speech.

The sf_std v/s sc_std and scr_std v/s zcr_std gives some seperation. But it would be hard to classify just with them as we can see, there are a lot of overlapping datapoints and it is hard to clearly draw a seperating boundary.

For sf_std v/s sc_std plot, sc_std is higher for speech probably because of presence of different syllables which create variations in the spectral centroids.

We dont think scr_mean vs sc_mean and sf_mean vs zcr_mean are good features pairs when used alone as they dont distinguish the data points so well.

Abbreviations:<br/>
sc : Spectral centroid<br/>
rms : Root mean square<br/>
zcr : Zero crossing rate<br/>
scr : Spectral crest<br/>
sf : Spectral flux<br/>
