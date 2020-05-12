# 3D EMOTION GAN: EMOTION-DRIVEN GAN AS AN INPUT METHOD FOR CREATING 3D ART FORMS
 by Kellyn Dassler

# Abstract
We propose a method for generating three dimensional objects from emotion-based two-dimensional image inputs via a two-step neural network process. Using an auxiliary classifier generative adversarial neural network framework with a 2D-to-3D style transfer neural network, we create novel three-dimensional objects driven by emotion-based art and user-provided emotion selection. Preliminary results show that our process generates objects that exhibit psychologically-based emotional features, and initial user studies reveal that this process may provide a new way for users to imbue personal emotions into 3D virtual objects and art.

# Introduction
Art psychologically reflects its creator’s emotions \cite{AlvarezMelis2017TheEG}. Similarly, art therapy, occupational therapy, and other mental health interventions utilize creative expression to regulate and improve emotional regulation \cite{article}. However, many of these interventions are cost-prohibitive, difficult for users to pick-up quickly, and only accessible to limited populations \cite{article}. Virtual art forms provide an artistic medium that addresses the aforementioned issues, but many users still feel intimidated by the creative process \cite{article}. While several studies have proposed methods for allowing users to generate two-dimensional images that reflect human emotions using generative adversarial networks \cite{AlvarezMelis2017TheEG}, none have studied methods for generating emotionally reflective three-dimensional forms. Thus, through this study, we seek new user access points to three-dimensional art creation via co-creative design techniques with artificially intelligent neural networks.

# Related Work

## A. GANs
Generative Adversarial Networks (GANs) were proposed in 2014 as a two network model to generate novel data from pre-existing data \cite{goodfellow2014generative}. They consist of at least one generator network and one discriminator network that are simultaneously trained in an adversarial fashion. For image data in particular, the generator creates increasingly improved novel data by testing generated images against the discriminator, which determines whether the generated image is ‘real’ or ‘fake’ \cite{goodfellow2014generative}.  As the discriminator is trained, its accuracy increases until the generator creates images that highly mirror the initial dataset \cite{goodfellow2014generative}. This flagship paper outlined the initial GAN framework, which has since been extended and improved upon in architectures like conditional GANs, collaborative GANs, and deep convolutional GANs  \cite{acgan}.

## B. Neural Style Transfer
Shortly after GANs debuted, Leon and colleagues proposed a method of transferring the art style of one image to the content of another image using biologically inspired deep neural networks \cite{Gatys_2016}. By separating intermediate layer activations that represent artistic features like edges, textures, color, and shape from a convolutional image classification network, network creators can generalize these features and apply them to any other image with novel content \cite{Gatys_2016}. Although this is a neural network-based approach, it is not generative in the true sense of creating completely novel outputs. 

## C. Emotion GAN
Created by MIT Media Lab members in 2017, the Emotional GAN uses a modified conditional GAN with an extensive WikiArt and MOMA artwork dataset to generate novel images that represent human emotions \cite{AlvarezMelis2017TheEG}. They took an independent approach to dataset creation by recruiting annotators to indicate the emotion most invoked in them by the image. They achieved state-of-the-art results for emotion-based image creation, and generated novel images in six different categories—anger, anxiety, fear, joy, sadness, and neutral. Although they created artwork that exhibited emotional features reflective of relevant psychological and artistic literature, they did not study the application of this framework to user creative expression.

## D. 3D Object Creation
Liu, Yu, and Funkhouser showed how novice users can create interactive three-dimensional shapes through a voxel model interface connected to an iterative GAN \cite{Liu_2017}. By applying a latent vector to a 3D voxel model created by the user alongside an iterative GAN and SNAP processing pipeline, users can generate a more realistic shape from the input \cite{Liu_2017}. Their research argues that GANs can provide a process through which users with no formal art training can participate in computer-assisted interactive modeling \cite{Liu_2017}, but it does not address creative forms of expression and human-GAN co-creation. 


# Methodology

# Two-Step Neural Network Emotion Transfer Process
 The emotion-to-three-dimensions framework was facilitated through a two-step neural network process. Initially, we trained an auxiliary classifier generative adversarial neural network (AC-GAN) using an emotion-labeled two-dimensional artwork dataset. After training, the AC-GAN was able to successfully create plausible novel artwork images reflecting major human emotions, including happiness, fear, anger, sadness, and anxiety. To use these images as novel input for creating three-dimensional objects, users were given a Jupyter notebook-based application through which they indicated their emotions and altered the generated photos before sending the photos to a 2D-to-3D neural transfer style network. In a preliminary user preference study, the participants were trained in both manual three-dimensional object creation using a polybrush in Unity3D as shown in Figure 1, and the two-step neural network object creation process as shown in Figure 2. After creating objects for each major emotion through both approaches, the users’ indicated their preferred creation method and produced outputs for each emotion. Finally, we interviewed each user to gather qualitative preference data to provide insight into user preferences. 
 
![Fig 1](images/figures/polybrushstart.png?raw=true "polybrushstart.png")
![Fig 2](images/figures/selection.jpg?raw=true "selection.jpg")
 
# Dataset Summary
The generated style images are rooted in an emotion-labeled artwork dataset processed from the original WikiArt artwork database and annotated with human emotions garnered from ten or more annotators, available from the WikiArt Emotions Dataset webpage \cite{LREC18-ArtEmo}. Each image was accompanied by floating point values that indicated the value of 20 different emotions present in the image. In total, 2865 annotated images were downloaded using support from the pandas Python package. To retrieve the major emotion most expressed by each image, we eliminated emotions that did not match one of five emotion categories-- happiness, fear, anger, sadness, and anxiety—and then processed the dataset to find the maximum value amongst the remaining emotion categories. These emotions were chosen due to their straightforward nature and consistency among users and high levels of representation within the dataset. Each image was then loaded into an image array and resized to 28x28 pixels, while each corresponding emotion label was processed into a label list to prepare the data for network training.

The content object files were created via object files using XCode with additional provided object files from the neural renderer repository. While we included objects other than spheres for future work, we only utilized the sphere object to run the study, in order to account for constraints and confounding variables. These were imported into the data examples directory for use by the transfer style neural network.

# ACGAN Structure
GANs are an artificially intelligent architecture for creating novel data from existing inputs \cite{goodfellow2014generative}. Conditional GANs, a deep convolutional subset of traditional GANs used for creating images, use class-labeled image input to create images of one or more chosen types \cite{10.5555/3305890.3305954}. The AC-GAN extends the generative process of the conditional GAN by adding a class label prediction into the discriminator network, which allows the network to create novel images and determine which class they fit into best \cite{10.5555/3305890.3305954}. This alteration stabilizes network training to produce higher quality image representations in the latent space \cite{acgan}. This structure is well-suited to our proposed method because it allows each generated output to be novel exhibit learned attributes indicative of a given emotion class. For more in-depth ACGAN information please see the associated report in Github.

![Fig 3](images/figures/ACGAN.png?raw=true "ACGAN.png")

# Neural Transfer 2D-to-3D Mesh Renderer
In contrast to traditional GANs, neural style transfer networks apply the features of a single image input to an unrelated output, rather than generating entirely novel data \cite{neuraltrans}. In the original style transfer algorithm, a style image is applied to a content image using a pretrained VGG19 network architecture for image classification, deconstructed into layers that are used to define the image contents and styles \cite{neuraltrans}. For our proposed method, we used a pre-trained 2D-to-3D mesh renderer network for image style transfer \cite{DBLP:journals/corr/abs-1711-07566}to three-dimensional objects called Neural Renderer \cite{DBLP:journals/corr/abs-1711-07566}. To convert an image into a polygon object mesh, the network utilizes an approximate gradient approach for rasterization to enable a rendering component in the network. Thus, the network can circumvent the usual prevention of backpropagation by rasterization and create three-dimensional mesh reconstructions that reflect the shape, color, and texture of the two-dimensional input image. According to the flagship neural renderer paper, this mesh reconstruction process outperforms existing voxel-based approaches \cite{DBLP:journals/corr/abs-1711-07566}.

# Network Training
We trained our AC-GAN model with Google CoLab’s TPU hardware, written in Python on a TensorFlow core using eighty percent of the class-labeled image data for a total of 2292 images. Each input image was pre-processed to 28x28 pixel images and the model generated novel image outputs for each emotion class— happiness, fear, anger, sadness, and anxiety. Following training, the images were validated and tested using the remaining twenty percent of image data. Generated images were analyzed for psychologically-based emotional feature representation, as described in the Emotion GAN paper \cite{AlvarezMelis2017TheEG}, and then displayed and stored for later use. 

For the next part of the study, a pre-trained 2D-to-3D neural renderer \cite{DBLP:journals/corr/abs-1711-07566} was used to convert the generated two-dimensional images into three-dimensional object meshes. This adapted code was built in Python and TensorFlow and used an approximate gradient for rasterization operations to prevent back-propagation \cite{DBLP:journals/corr/abs-1711-07566}. This network was run for each selected emotion during the user study using user-selected generated image data from the AC-GAN network as the style and a sphere object as the content. Each stylized object reflected the shape, color, and texture of the user-selected generated image. 

For more in-depth training information please see the associated report on Github.

# User Preference Study
To test our proposed creative method, we conducted a user preference study among six participants. The study was comprised of two creation input conditions: a Unity3D polybrush virtual object creation process (hereafter referred to as the polybrush process) and our three-dimensional emotion GAN virtual object creation process (hereafter referred to as the emotion GAN process). Each participant created objects through both processes for each of the five emotions---- happiness, fear, anger, sadness, and anxiety. The process creation order was randomized for each user. To reduce confounding variables, none of the selected participants had previous experience with Unity3D or artificial intelligence co-creation techniques. Additionally, each participant was given an initial questionnaire that asked about their comfort levels with art creation, how strongly they felt their emotions, and whether or not they tend to express their emotions through creative outlets such as art, music, or writing.

For the polybrush process, the participants were trained in polybrush and Unity techniques, and the polybrush tool was set to utilize open edges and a sticky brush technique among all users. They were allowed to alter shape, texture, color, and smoothing of their given structure. After practicing polybrush maneuvers, the users were given a sphere-shaped virtual object in the Unity platform and were asked to produce a three-dimensional object that reflected their understanding of one of the five selected major human emotions. The order of each emotion was randomized for each participant. The participant was then given 10 minutes to create and finish their virtual sculpture. This was repeated for each of the five emotions, and each virtual object was stored to an object file for later review.

For the emotion GAN process, each participant was provided with a user-friendly Jupyter notebook that featured drop-down widgets to make emotion and image selections. They were each given a brief overview of the idea that computers can create novel image data, the two-step process for generating a novel object, and were trained in using the buttons and widgets in the Jupyter notebook. During the process, users were asked to click a button that reflected images from one of the five major emotions. Then, they were provided with at least five photos generated by the pre-trained emotion images network previously described. The participants were then asked to select one of the five photos as the “most representative” of the given emotion. This selected image was then applied to a virtual sphere object using the neural renderer and downloaded as an object file. As in the polybrush process, the users were asked to produce a virtual object for each emotion in a randomized fashion through this method, and each object file was saved for later review.  

After both processes were completed, each participant was shown their virtual objects created through the polybrush process and the emotion GAN process side-by-side for each emotion. The participant was asked to rate their virtual object preference for each emotion, and for all the emotions overall. Then, they were asked which input process they preferred. The results were recorded and analyzed.

# Results
After each of the six user rounds were completed, the results were compiled and analyzed. An example of a polybrush generated sculpture versus an emotion GAN generated sculpture for "angry" can be seen in Figure 4. Overall, 4 out of 6 participants, or sixty-seven percent, preferred the polybrush created objects over the emotion GAN created objects. However, for certain emotions, including anxiety and fear, all but one participant preferred the emotion GAN created objects. Summaries of these results can be seen in Figure 5. Interestingly, despite preferring the polybrush created objects over the emotion GAN created objects overall, all but one participant also preferred the emotion GAN process to the polybrush process. 

![Fig 4.1](images/figures/angryobject.png?raw=true "angryobject.png")

![Fig 4.2](images/figures/angryGAN.png?raw=true "angryGAN.png")

![Fig 5.1](images/figures/emotiongraph.png?raw=true "emotiongraph.png")

![Fig 5](images/figures/overallgraph.png?raw=true "overallgraph.png")

# Discussion
Due to study restraints resulting from COVID-19 restrictions, we could only run the study with six participants. Thus, we were unable to provide statistically valid quantitative analysis for the study with \emph N = 6. However, to better understand the results, we conducted participant post-study surveys to garner reasoning and starting points for future studies. 

Many of the participants indicated that they preferred the polybrush created objects to the emotion GAN created ones, because they felt as if they had more control over the outcome. As one user said, “I like to make small adjustments and add lots of different colors, rather than just having the computer decide for me.” When asked about their simultaneous preference for the emotion GAN creation process, however, most participants said that it was simpler to use than the polybrush tool, and they liked the new possibilities that the GAN might create. One user also suggested that for more complex emotions like anxiety and fear, the emotion GAN also helped her “come up with a starting point” and the artistic renditions “did a better job of expressing [these] complex emotions, since [she] didn’t really know how to express that”. This user also indicated in her pre-survey that she did not usually express her emotions strongly, nor did she usually express them through creative practices.

These seemingly conflicting opinions offer some insight into the artificial intelligence co-creation process. Perhaps users would prefer more options for adjusting the final outcome of the generated image via easy-to-use interactive shaping tools or widgets, while still using the emotion GAN generated image as a starting point for creation. Similarly, offering choices for different object shapes in the emotion GAN process may provide a way for more user control in shaping the final object outcome. These may provide a basis for future studies on improving this process, but the results still suggest that our proposed generative method may provide a more accessible gateway to three-dimensional emotion-based art for many users. 

# Conclusion
In this study, we have proposed a novel two-step method for generating emotion-based three-dimensional artwork from two-dimensional images and user input. Although we did not achieve state-of-the-art results, and most users preferred more control over the final representation of their generated objects, we did provide a preferred gateway for users to engage in co-creative design with artificially intelligent networks. By improving upon this process via more user input inclusion, better image generation, and three-dimensional object options, future researchers can iterate upon a viable co-creative design process for generating emotion-based three-dimensional virtual objects for creative expression and therapeutic purposes.

# Future Work
In future studies, we suggest several alterations for improving and testing our proposed generative method. During the study, we only had access to a limited dataset, which resulted in less than state-of-the-art image generations from the ACGAN. As with most generative networks, the emotion based ACGAN will likely exhibit improved results if trained with larger amounts of more robust data. While we only tested five emotions separately, the neural network can easily be adjusted to generate images that represent several different emotions, or layers of emotions. For example, future research may want to test user preferences for creating objects that reflect both anger and sadness, or more complex emotions like confusion, regret, or shame. Additionally, we recommend that future researchers generate more “adjustment widgets” and object types to test user preferences for control and input during the emotion GAN process. Finally, we propose using facial recognition networks to test using facial emotions as an input method for the generative emotion GAN process, rather than user button selected emotions. While this was initially a proposed method for the study, we could not complete it due to time and situation constraints. However, we think this is a valuable avenue for generating novel input methods for the virtual creative process. These changes will generate new understandings about the human-artificial intelligence co-creative process.

# Acknowledgements
Special thanks to Francisco for being a very intelligent, encouraging professor and research mentor!

# References
@inproceedings{AlvarezMelis2017TheEG,
  title={The Emotional GAN : Priming Adversarial Generation of Art with Emotion},
  author={David Alvarez-Melis},
  year={2017}
}
@article{article,
author = {Brinck, Ingar and Reddy, Vasudevi},
year = {2019},
month = {07},
pages = {},
title = {Dialogue in the making: emotional engagement with materials},
journal = {Phenomenology and the Cognitive Sciences},
doi = {10.1007/s11097-019-09629-2}
}
@online{acgan,
  author = {Brownlee, Jason},
  title = {How to Develop an Auxiliary Classifier GAN (AC-GAN) From Scratch with Keras},
  year = 2020,
  url = {https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/},
  urldate = {2020-03-23}
}
@article{Gatys_2016,
   title={A Neural Algorithm of Artistic Style},
   volume={16},
   ISSN={1534-7362},
   url={http://dx.doi.org/10.1167/16.12.326},
   DOI={10.1167/16.12.326},
   number={12},
   journal={Journal of Vision},
   publisher={Association for Research in Vision and Ophthalmology (ARVO)},
   author={Gatys, Leon and Ecker, Alexander and Bethge, Matthias},
   year={2016},
   month={Sep},
   pages={326}
}
@misc{goodfellow2014generative,
    title={Generative Adversarial Networks},
    author={Ian J. Goodfellow and Jean Pouget-Abadie and Mehdi Mirza and Bing Xu and David Warde-Farley and Sherjil Ozair and Aaron Courville and Yoshua Bengio},
    year={2014},
    eprint={1406.2661},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
@article{DBLP:journals/corr/abs-1711-07566,
  author    = {Hiroharu Kato and
               Yoshitaka Ushiku and
               Tatsuya Harada},
  title     = {Neural 3D Mesh Renderer},
  journal   = {CoRR},
  volume    = {abs/1711.07566},
  year      = {2017},
  url       = {http://arxiv.org/abs/1711.07566},
  archivePrefix = {arXiv},
  eprint    = {1711.07566},
  timestamp = {Mon, 13 Aug 2018 16:48:37 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1711-07566.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
@online{neuraltrans,
  title = {How to Develop an Auxiliary Classifier GAN (AC-GAN) From Scratch with Keras},
  year = 2020,
  url = {https://www.tensorflow.org/tutorials/generative/style_transfer},
  urldate = {2020-03-23}
}
@article{Liu_2017,
   title={Interactive 3D Modeling with a Generative Adversarial Network},
   ISBN={9781538626108},
   url={http://dx.doi.org/10.1109/3DV.2017.00024},
   DOI={10.1109/3dv.2017.00024},
   journal={2017 International Conference on 3D Vision (3DV)},
   publisher={IEEE},
   author={Liu, Jerry and Yu, Fisher and Funkhouser, Thomas},
   year={2017},
   month={Oct}
}
@inproceedings{LREC18-ArtEmo,
    author = {Mohammad, Saif M. and Kiritchenko, Svetlana},
    title = {An Annotated Dataset of Emotions Evoked by Art},
    booktitle = {Proceedings of the 11th Edition of the Language Resources and Evaluation Conference (LREC-2018)},
    year = {2018},
    address={Miyazaki, Japan}
}
@inproceedings{10.5555/3305890.3305954,
author = {Odena, Augustus and Olah, Christopher and Shlens, Jonathon},
title = {Conditional Image Synthesis with Auxiliary Classifier GANs},
year = {2017},
publisher = {JMLR.org},
booktitle = {Proceedings of the 34th International Conference on Machine Learning - Volume 70},
pages = {2642–2651},
numpages = {10},
location = {Sydney, NSW, Australia},
series = {ICML’17}
}
@online{polybrush,
  title = {Polybrush Introduction and Tutorial},
  year = 2020,
  url = {https://unity3d.com/unity/features/worldbuilding/polybrush},
  urldate = {2020-03-23}
}
