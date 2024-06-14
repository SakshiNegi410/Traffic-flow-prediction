# Traffic-flow-prediction
Traffic prediction is the task of predicting future traffic measurements (e.g. volume, speed, etc.) in a road network (graph), using historical data (timeseries) 
1.INTRODUCTION 
1.1. INTRODUCTION & OBJECTIVE 
Traffic congestion can have substantial effects on quality of life, especially in bigger cities. It is estimated that traffic congestion in United States causes two billion gallons of fuel to be wasted every year; 135 million US drivers spend two billion hours stuck in traffic every year. Altogether, 100 billion USD are spent because of fuel in the US alone. For an average American driver, this costs 800 USD per year (Liu et al., 2006). In addition to economic aspect (wasting money and time), there is also an ecological one. Pollution could be reduced significantly by reducing travel time and thus emissions. The above-mentioned facts are the main reasons that governments are investing in Intelligent Transportation Systems (ITS) technologies that would lead to more efficient use of transportation networks. Traffic prediction models have become a main component of most ITS. Accurate real time information and traffic flow prediction are crucial components of such systems. ITS vary in technologies applied in; from advanced travel information systems, variable message signs, traffic signal control systems, to special user-friendly applications, such as travel advisors. The aim of all of these technologies is the same, to ease traffic flow, reduce traffic congestion and decrease travel time by advising drivers about their routes, time of departure, or even type of transportation. The availability of large amounts of traffic related data, collected from a variety of sources and emerging field of sophisticated machine learning algorithms, has led to significant leap from analytical modelling to data driven modelling approach. The main concept of this paper is to investigate different machine learning algorithms and engineer features that would enable us predicting traffic state several prediction intervals into the future
In general, traffic prediction studies can be categorized into three major categories: naïve methods, parametric methods and non-parametric methods. Naïve methods are usually simple non-model baseline predictors, which can sometimes return good results. Parametric models are based on traffic flow theory and are researched separately and in parallel to non-parametric, more data-driven machine learning methods. A strong movement towards non-parametric methods can be observed in the recent years, probably because of increased data availability, the progress of computational power and the development of more sophisticated algorithms. Non-parametric does not mean models are without parameters; but refers to model’s parameters, which are flexible and not fixed in advance. The model’s structure as well as model parameters are derived from data. One significant advantage of this approach is that less domain knowledge is required in comparison to parametric methods, but also more data is required to determine a model. This also implies that successful implementation of data-driven models is highly correlated to the quality of available data. In contrast, traffic in urban areas can be much more dynamic and non-linear, mainly because of the presence of many intersections and traffic signs. In such environments, data-driven machine-learning approaches, such as neural Networks, Random Forests and kNN, can be more appropriate, due to their ability to model highly nonlinear relationships and dynamic processes.
1.2.	PURPOSE OF THE PROJECT 
Its purpose is to build a Machine Learning web application which will help and guide to find a easiest way to find less traffic route and reduce the number of accidents.
1.3.	EXISTING SYSTEM & DISADVANTAGES 
       Vehicle identification is an important technique in the transport system. By identifying the vehicle, the number of automobiles is known and the presences of the vehicles on the path are important factors. High dimensional data can be used to denote the automobiles. Feature retrieval and classifying features are the important processes used to identify the automobiles. High dimension data takes more computation time during the feature extraction process. D. M. S. Arsa et al., 2017, proposes the DBN (Deep Belief Network) technique for reducing the dimension of the data to detect the vehicles. In this research, the authors try to identify motorbikes and cars. Here DBN technique is used to reduce the dimension of the data and the SVM concept is used to classify the data. The proposed method applied to the UIUC dataset and the outcome of the current technique is compared with the PCA method. The experiment outcomes show that DBN concept provides a better result than traditional PCA method in the identification of automobiles
In the transport system traffic flow forecasting is a major issue. Various existing techniques produce unsuccessful output due to various reasons like thin framework, engineering manually, and learning separately. W. Huang et al., 2014 proposes a new deep framework that contains two basic parts. At the base part contains DBN and the top portion contains the regression layer. DBN technique is applied for the purposed of unsupervised learning and it learns efficient attributes for traffic forecasting an unsupervised manner. It produces a better result for many places like audio and image classification. The regression layer is used for supervised forecasting. The experiment outcomes describe that the proposed method increases the performance of existing systems. The positive outcomes say that multitask regression and deep learning are important technologies in the transport system research



1.4.	PROPOSED SYSTEM & ITS ADVANTAGES 
In this project we will be exploring the datasets of four junctions and built a model to predict the traffic on the same. This could potentially help in solving the traffic congestion problem by providing a better understanding of traffic patterns that will further help in building an infrastructure to eliminate the problem. We used different ML algorithms like LSTM, GRU, Random Forest Regressor, KNN to get a more accurate prediction result 
2.SYSTEM ANALYSIS
2.1 STUDY OF THE SYSTEM 
                  A survey of the recent literature suggests that many authors have contributed their well enough in the field of traffic Incident analysis, prediction and their relation in connection with the traffic congestions. A simple visualization-based approach to show traffic incidents from the past data as map overlay in the form of dynamic radial circles has been given in [9]. Traffic Origins are different colored circles, each representing different road conditions i.e. heavy traffic, breakdowns and the congestions that are plotted on the map [9]. The traffic origins are the visual descriptors of the location of the incident, heavy traffic flow and the breakdowns whereas their radius determine the vicinity in which the traffic would be affected in one way or the other [9]. Once the area is cleared the circle recedes back and eventually vanishes at the central point of their origin [9]. According to [9], the traffic origin visualization technique helps better in determining the effects that a cascaded accident or constricted traffic flow could potentially have on a particular road in a traffic network. From literature review traffic flows forecasting can be broadly classified into two distinct categories, Parametric approaches that are based on statistical methods for time series forecasting. Knowledge of data distributions are usually assumed in these approaches. These traffic process-based prediction methods mostly employ traffic systems simulations, road activities and driver behavior parameters as part of the simulation process. The macroscopic traffic prediction models are based on the vehicular traffic flow analogies with fluid dynamics [10]. The major advantage of using the macroscopic simulations for traffic predictions is that in such methods traffic control parameters (e.g. delay at traffic lights, average time spent on bus stops etc.) can be used in the predictions process and the better understanding of the real traffic environment is achieved based on the locations. On the other hand, the disadvantage of 21 using much macroscopic prediction techniques are the complex parameter estimations and a real struggle to generate close to real world simulation test environment. Also, the predictions are highly influenced by the quality of the estimated traffic parameters [11]. Both the statistical ML and macroscopic approaches are useful for the ideal traffic flow prediction model development. This research however, focuses purely on the study for the data driven statistical to complex ML methods for traffic related predictions. The major difference between the ML and conventional analytical method-based models is that ML is considered as a black box which learns the relationships between the inputs and the outputs to predict traffic variables. While ML models are complicated to optimize for the learning, but they are less complicated and computationally efficient to calculate the final prediction once trained. Continued training allows ML models to adapt to the changing behavior displayed in the data. A detailed description of the selected models can be found in the section 4. According to the literature, for comparison to be meaningful the same traffic data needs to be used for both the statistical and computational learning models. However, such a comparison of models across the literature with same data used in different comparison scenarios is difficult to be found.
2.2 INPUTcAND OUTPUT DESIGN
2.2.1 INPUT DESIGN
Thecinput design is theclink between the informationcsystem and the user. It comprises thecdeveloping specification andcprocedures for data preparationcand those steps are necessary to putctransaction data in to acusable form for processing cancbe achieved by inspecting theccomputer to read data fromca written or printed documentcor it can occur by having peopleckeying the data directlycinto the system. The designcof input focusescon controlling thecamount of input required, controllingcthe errors, avoiding delaycavoiding extra steps and keepingcthe process simple. The inputcis designed in such acway so that it provides security andcease of use with retaining thecprivacy. Input Design consideredcthe following things:

	What datacshould be given ascinput?
	How thecdata should be arranged orccoded?
	The dialogcto guide the operatingcpersonnel in providing input.
	Methods forcpreparing input validationscand steps to follow when errorcoccur.

2.2.2 OBJECTIVES
1.InputcDesign is the process ofcconverting a user-orientedcdescription of the input into a computer-basedcsystem. This design iscimportant to avoid errorscin the data input process and show the correctcdirection to the managementcfor getting correct informationcfrom the computerized system.
                  2.Itcis achieved by creatingcuser-friendly screens for thecdata entry to handle large volume ofcdata. The goal of designingcinput is to make data entryceasier and to be freecfrom errors. The datacentry screen is designed incsuch a way that all thecdata manipulates can be performed. Itcalso provides record viewingcfacilities.
3.Whenthecdata is entered it will checkcfor its validity. Data can becentered with the help of screens. Appropriatecmessages are provided ascwhen needed so that thecuser will not be in maize ofcinstant. Thus the objective ofcinput design is to createcan input layout that is easycto follow

2.2.3 OUTPUT DESIGN
                     A qualitycoutput is one, which meetscthe requirements of the endcuser and presents the informationcclearly. In any systemcresults of processing areccommunicated to the userscand to other systemcthrough outputs. In outputcdesign it is determined howcthe information is tocbe displaced forcimmediate need andcalso the hard copycoutput. It is the mostcimportant and direct source informationcto the user. Efficientcand intelligent outputcdesign improves the system’s relationshipcto help user decisioncmaking.
                     1. Designingccomputer output shouldcproceed in an organized, wellcthought out manner; the rightcoutput must be developedcwhile ensuring that eachcoutput element iscdesigned so that peoplecwill find the systemccan use easily andceffectively. When analysiscdesign computer output, theycshould Identify the specificcoutput that is needed to meetcthe requirements.
2.Selectcmethods for presentingcinformation.
3.Createcdocument, report, or othercformats that contain informationcproduced by the system.
Thecoutput form of an informationcsystem should accomplish onecor more of the followingcobjectives.
•	Conveycinformation about pastcactivities, current status orcprojections of the
•	Future.
•	Signalcimportant events, opportunitiescproblems, or warnings.
•	Triggercan action.
•	Confirmcan action.
2.3 PROCESS MODEL USED WITH JUSTIFICATION SDLC (Umbrella Model):  
     SDLC IS NOTHING BUT SOFTWORE LIFE CYCLE
Stages in SDLC: 
♦ Requirement Gathering  
♦ Analysis  
♦ Designing  
♦ Coding  
♦ Testing  
♦ Maintenance  
Requirements Gathering stage: 
The requirements gathering process takes as its input the goals identified in the high-level requirements section of the project plan. Each goal will be refined into a set of one or more requirements. These requirements define the major functions of the intended application, define  operational data areas and reference data areas, and define the initial data entities. Major functions include critical processes to be managed, as well as mission critical inputs, outputs and reports. A user class hierarchy is developed and associated with these major functions, data areas, and data entities. Each of these definitions is termed a Requirement. Requirements are identified by unique requirement identifiers and, at minimum, contain a requirement title and textual description.  
Analysis Stage: 
The planning stage establishes a bird's eye view of the intended software product, and uses this to establish the basic project structure, evaluate feasibility and risks associated with the project, and describe appropriate management and technical approaches.  
The most critical section of the project plan is a listing of high-level product requirements, also referred to as goals. All of the software product requirements to be developed during the requirements definition stage flow from one or more of these goals. The minimum information for each goal consists of a title and textual description, although additional information and references to external documents may be included. The outputs of the project planning stage are the configuration management plan, the quality assurance plan, and the project plan and schedule, with a detailed listing of scheduled activities for the upcoming Requirements stage, and high-level estimates of effort for the out stages.  




Designing Stage: 
The design stage takes as its initial input the requirements identified in the approved requirements document. For each requirement, a set of one or more design elements will be When the design document is finalized and accepted, the RTM is updated to show that each design element is formally associated with a specific requirement. The outputs of the design stage are the design document, an updated RTM, and an updated project plan.  
Development (Coding) Stage: 
The development stage takes as its primary input the design elements described in the approved design document. For each design element, a set of one or more software artifacts will be produced. Software artifacts include but are not limited to menus, dialogs, data management forms, data reporting formats, and specialized procedures and functions. Appropriate test cases will be developed for each set of functionally related software artifacts, and an online help system will be developed to guide users in their interactions with the software.  
Integration & Test Stage: 
During the integration and test stage, the software artifacts, online help, and test data are migrated from the development environment to a separate test environment. At this point, all test cases are run to verify the correctness and completeness of the software. Successful execution of the test suite confirms a robust and complete migration capability. During this stage, reference data is finalized for production use and production users are identified and linked to their appropriate roles. The final reference data (or links to reference data source files) and production user list are compiled into the Production Initiation Plan.  



♦ Installation & Acceptance Test: 
During the installation and acceptance stage, the software artifacts, online help, and initial production data are loaded onto the production server. At this point, all test cases are run to verify the correctness and completeness of the software. Successful execution of the test suite is a prerequisite to acceptance of the software by the customer.  
After customer personnel have verified that the initial production data load is correct and the test suite has been executed with satisfactory results, the customer formally accepts the delivery of the software.  
Maintenance: 
Outer rectangle represents maintenance of a project, Maintenance team will start with requirement study, understanding of documentation later employees will be assigned work and they will under go training on that particular assigned category.  For this life cycle there is no end, it will be continued so on like an umbrella (no ending point to umbrella sticks).  
2.4. SYSTEM ARCHITECTURE 
Architecture flow: 
Below architecture diagram represents mainly flow of the application from data input to prediction result generation
 ![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/e2c1c8c0-21f7-4d98-af10-2066d74d71a5)

2.5FEASIBILITY STUDY
Believabilityis the determination of paying littlerespect to whether anundertaking justifies action. The frameworkfollowed in building theirstrength is called acceptabilityStudy, these kinds of studyif a task could andought to be taken.
Three keythoughts included inthe likelihood examinationare:
	TechnicalFeasibility
	EconomicFeasibility
	OperationalFeasibility

2.5.1 TechnicalFeasibility
Here it isconsidered with determininghardware and programming, thiswill effective fulfill the client necessity the specialized requiresof the framework shouldshift significantly yetmay incorporate
	Theoffice to create yields in specified time.              
	 Reaction timecunderparticularcstates.
	 Capacity todeal with a particular segmentof exchange at a specific pace.

2.5.2 EconomicFeasibility
Budgetaryexamination is the oftenused system for assessing thefeasibility of a projected structure. Thisis more usually acknowledgedas cost/favorablepositionexamination. The method is tocenter the focal points andtrusts are typical casing aprojected structure and a difference themand charges. These pointsof interest surpass costs; achoice is engaged to diagram andrealize the system willmust be prepared if thereis to have a probability ofbeing embraced. There is aconsistent attempt thatupgrades in exactness atcall time of the systemlife cycle.







2.5.3 OperationalFeasibility
It is for themost part identified withhuman association andsupporting angles. The focusesare considered:
What alterationscwill be carried through thecframework?
•	Whatcauthoritative shapes arecdispersed?
•	What newcaptitudes will becneeded?
•	Do theccurrent framework employee’scindividuals have thesecaptitudes?
•	If not, wouldcthey be able to becprepared over the spancof time?
















 4.REQUIREMENT SPECIFICATION                                           
4.1. FUNCTIONAL REQUIREMENTS  
There are three modules can be divided here for this project they are listed as below
•	User input form for traffic flow calculation
•	Best route suggestion as per user input
•	Traffic flow pattern graph generation
•	Display predicted results/output
From the above four modules, project is implemented. Bag of discriminative words are achieved
•	User input form for traffic flow calculation:
The data can be entered by admin without any particular scenario but with the details of features data. The most importantly large amount of can be handled in order to do practically. The data that are handling throughout the project can be done in this module. Users have permission to view data but not edit the data in online they can request the user to get the data.
•	Best route suggestion as per user input:
The system will suggest the user having less traffic as per the input given by user
•	Traffic flow pattern graph generation:
This module will show the daily traffic up and down as per vehicle and time ratio
•	Display Predicted Results/Outputs:
The prediction result will be displayed on the webpage to the user.

4.2	PERFORMANCE REQUIREMENTS 
Performance is measured in terms of the output provided by the application. Requirement specification plays an important part in the analysis of a system. Only when the requirement specifications are properly given, it is possible to design a system, which will fit into required environment. It rests largely with the users of the existing system to give the requirement specifications because they are the people who finally use the system. This is because the requirements have to be known during the initial stages so that the system can be designed  



4.3	SOFTWARE REQUIREMENTS:  
Operating System : Windows  
Technology : python 3.8.0, Tensorflow , scikit-learn
Web Technologies : Flask framework,Html, JavaScript, CSS  
Web Server : Flask server
Software’s : Python, VS code
4.4	HARDWARE REQUIREMENTS:  
Hardware : Pentium based systems with a minimum of P4  	 RAM : 256MB (minimum) 
Additional Tools: 
HTML Designing : Dream weaver Tool  
Development Tool kit : VS Code

4.5 TECHNOLOGY USED
4.5.1 PYTHON
Pythoncis a general-purposecinterpreted, interactive, objectcoriented, and high-level programmingclanguage. An interpreted languagecPython has a design philosophycthat emphasizes codecreadability (notably usingcwhitespace indentation to delimit codecblocks rather than curly bracketscor keywords), and a syntaxcthat allows programmers to expresscconcepts in fewer lines of codecthan might be used in languagescsuch as C++or Java. It providescconstructs that enable clearcprogramming on both smallcand large scales.Pythoncinterpreters are available for many operatingcsystems. CPython, the reference implementationcof Python, is open source softwarecand has a community-basedcdevelopment model, as docnearly all of its variant implementations. C Pythoncis managed by thecnon-profit Python SoftwarecFoundation. Python features acdynamic type system andcautomatic memory management. Itcsupportsmultiplecprogramming paradigms, includingcobject-oriented, imperativecfunctional and procedural, and hasca large and comprehensivecstandard library



4.5.2 FLASK
Flaskcis a high-level Python Webcframework that encouragescrapid development and clean, pragmaticcdesign. Built by experiencedcdevelopers, it takes care of muchcof the hassle of Web development, socyou can focus on writingcyour app without needingcto reinvent the wheel. It’s free andcopen source.
Flask'scprimary goal is to ease theccreation of complex, databasecdriven websites. Flask emphasizescreusabilityand "pluggability" of componentscrapid development, andcthe principle of don't repeatcyourself. Python is usedcthroughout, even for settingscfiles and datacmodels. 
![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/e3162540-6c25-4894-94d4-993f76b85bc0)

 
Flask alsocprovides an optionalcadministrative create, readcupdate and delete interfacecthat is generated dynamicallycthrough introspectioncand configured via admincmodels

 ![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/ab7b9c5f-0003-4f99-a774-01deb15e4efc)















5.SYSTEM DESIGN
5.	1.INTRODUCTION 
Systems design 
Introduction: Systems design is the process or art of defining the architecture, components, modules, interfaces, and data for a system to satisfy specified requirements. One could see it as the application of systems theory to product development. There is some overlap and synergy with the disciplines of systems analysis, systems architecture and systems engineering.  
5.2Data Flow Diagrams  
Data flow diagram will act as a graphical representation of the system in terms of interaction between the system, external entities, and process and how data stored in certain location.  
•	External entities  
•	Data stores  
•	Process  
•	Data Flow 
![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/89222f64-99bc-465c-9131-de0440037288)

 
5.3 UML DIAGRAMS 
Unified Modeling Language:  
The Unified Modeling Language allows the software engineer to express an analysis model using the modeling notation that is governed by a set of syntactic semantic and pragmatic rules.  
A UML system is represented using five different views that describe the system from distinctly different perspective. Each view is defined by a set of diagram, which is as follows.  
 ![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/1fad5577-d497-43f0-9cb3-a328d7b079e8)

•	User Model View  
i.	This view represents the system from the users perspective.   
ii.	The analysis representation describes a usage scenario from the end-users 
perspective.  
•	Structural model view  
i. In this model the data and functionality are arrived from inside the 
system. ii. This model view models the static structures.  

•	Behavioral Model View  
It represents the dynamic of behavioral as parts of the system, depicting the interactions of collection between various structural elements described in the user model and structural model view.  

•	Implementation Model View  
In this the structural and behavioral as parts of the system are represented as they are to be built.  
•	Environmental Model View  
In this the structural and behavioral aspects of the environment in which the system is to be implemented are represented.  
UML is specifically constructed through two different domains they are:  
􀂃 UML Analysis modeling, this focuses on the user model and structural model views of the system.  
􀂃 UML design modeling, which focuses on the behavioralmodeling, implementation modeling and environmental model views.  
Use case Diagrams represent the functionality of the system from a user’s point of view. Use cases are used during requirements elicitation and analysis to represent the functionality of the system. Use cases focus on the behavior of the system from external point of view.  
Actors are external entities that interact with the system. Examples of actors include users like administrator, bank customer …etc., or another system like central database. 





5.4. ER DIAGRAMS 
 




![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/8e5930e1-6b86-42b9-a819-b52f797b339c)





6.METHOD&WORKING STEPS
STEP-1: IMPORTING LIBRARIES
 
![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/3297cf69-bad5-439f-adc6-413d516865d8)

STEP-2: LOADING DATA

 

![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/01236f56-8250-4011-8e42-2db93900ef12)





About the data
This dataset is a collection of numbers of vehicles at four junctions at an hourly frequency. The CSV file provides four features:
•	DateTime
•	Junctions
•	Vehicles
•	ID
The sensors on each of these junctions were collecting data at different times, hence the traffic data from different time periods. Some of the junctions have provided limited or sparse data.

STEP-3: DATA EXPLORATION
•	Phrasing dates
•	Plotting timeseries
•	Feature engineering for EDA

![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/43e3b5fe-1a08-4a6e-8f28-6b03e8981b03)




 ![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/e19760b3-2958-488a-a156-98747c8ab347)



 


 
![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/ab24b084-4acb-4105-9a7f-3da0d9874e70)





 


Noticeable information in the above plot:
•	It can be seen here that the first junction is visibly having an upward trend.
•	The data for the fourth junction is sparse starting only after 2017
•	Seasonality is not evident from the above plot, So we must explore datetime composition to figure out more about it.
STEP-4: FEATURE ENGINEERING

 ![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/4710d3e6-561e-4956-85b7-887b08ccdb13)



 ![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/13ee1f0e-b6b6-4d03-a007-40012ac5ca00)


STEP-5: EXPLORATORY DATA ANALYSIS

 ![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/7e9e7e52-efcd-45e9-be5a-b84d8bbee9c1)


 ![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/3124aac4-3e18-470f-b257-6cdbb0a6a347)


From the above plot following things can be concluded:
•	Yearly, there has been an upward trend for all junctions except for the fourth junction. As we already established above that the fourth junction has limited data and that don't span over a year.
•	We can see that there is an influx in the first and second junctions around June. I presume this may be due to summer break and activities around the same.
•	Monthly, throughout all the dates there is a good consistency in data.
•	For a day, we can see that are peaks during morning and evening times and a decline during night hours. This is as per expectation.
•	For weekly patterns, Sundays enjoy smoother traffic as there are lesser vehicles on roads. Whereas Monday to Friday the traffic is steady.
STEP-6: MODEL BUILDING

 ![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/06ede2e6-430a-4e17-afa5-0c28e866a6da)


 


STEP-7: FITTING THEMODEL

Fitting the first junction and plotting the predictions and testset
 

 




![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/9a76cc49-ccbd-45b5-a954-5e18d5aa9a6d)


Fitting the second junction and plotting the predictions and testset

 

 
![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/a12dc9ed-cde8-4d69-afd6-2f26649b1499)


Fitting the third junction and plotting the predictions and testset
 


 
![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/532dd7cc-d37b-4c6f-a289-d795213a56a9)





![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/df157d8d-2cba-4016-9408-0e3cf0c1c811)






ALGORITHM
Convolutional Neural Networks 

In image analysis, many of recent advances in deep learning are built on the work of LeCun et al. [18] who introduced a Convolutional Neural Network (CNN) which had a large impact on the eld. A CNN is a type of a neural network that is designed to process an image and represent it with a vector code. The architecture of CNNa draws on fully-connected neural networks. Similarly, a convolutional neural network is a compounded structure of several layers processing signals and propagating them forward. 


However, in contrast to a vector activation in a fully-connected layer, activations in CNNs have a shape of three-dimensional tensors. Commonly, this output tensor is called a feature map. For instance, an input image of shape 3 W H is transformed by the rst convolutional layer into a feature map of shape C W 0 H0, where C is the number of features. In other words, a convolutional layer transforms a volume into a volume. 


A typical CNN consists of several convolutional layers and, at the top, fully con-nected layers that atten convolutional volumes into a vector output. In the eld's terminology, this vector code of an image is often called fc7 features as it used to be extracted from the seventh fully connected layer of AlexNet [2]. Even though AlexNet has already been outperformed by many and the state-of-the-art designs are di erent from AlexNet, the term maintained its popularity. Additionally, depending on a prob-lem the network is supposed to solve, an additional layer, such as soft-max, can be added on top of fc7 features. A common design of a CNN is depicted in Fig. 4 

Receptive Field 

 
![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/a7b1b1de-eb16-438b-bdd2-178c15e47018)





As mentioned above, a convolutional layer takes a tensor on input and produces a ten-sor, too. Note that these tensors have two spatial dimensions W and H, and one feature dimension C as they copy the form images are stored in. The context conveyed by the spatial dimensions is utilized in the CNN design which takes into account correlations in small areas of the input tensor called receptive elds. Concretely, in contrast to a neuron in a fully connected layer that processes all activations of the previous layer, a neuron in a convolutional layer "sees" only activations in its receptive eld. Instead of transforming layer's activations it restraints to a speci c small rectangularly shaped subset of the activations. When mentioning a receptive eld, it is often expected only spatial dimensions of the input volume are referred to, i.e. a receptive eld de nes an area in the W H grid. The shape of the receptive eld is a hyperparameter and varies across the models. 


Figure 4: A convolutional neural network takes an image on input (in blue) and transforms it into a vector code (in blue). Convolutional Neural Networks are characteristic for processing volumes. An output of each layer is illustrated as an orange volume. Each neuron process only activations in the previous layer that belong to its receptive eld. The same set of weights is used for neurons across the whole grid. On top of convolutional layers, fully-connected layers are commonly connected. 



Convolution in CNNs 

A neuron's receptive eld is processed similarly to fully connected layer neurons. The values below the receptive eld along the input tensor's full depth are transformed by a non-linear function, typically ReLU (Eq. (8)). 

However, in contrast to fully connected layer neurons, the same set of weights (referred to as a kernel) is used for all receptive elds in the input volume resulting into a transformation that has a form of convolution across the input. A kernel is convolved across W and H spatial dimensions. Then, a di erent kernel is again convolved across the input volume producing another 2D tensor. Aligning up the output tensors into a C W 0 H0 volume assembles the layer's output feature map. 

This is an important property of convolutional neural networks because each kernel detects a speci c feature in the input. For example, in the rst layer, the rst kernel would detect presence of horizontal lines in the receptive elds, the second kernel would look for vertical lines, and similarly further on. In fact, learning such types of detectors in the bottom layers is typical for 
CNNs. 

  The design of CNNs has an immensely practical implication { since a kernel is con-volved across the input utilizing the same set of weights and it covers only the receptive eld, the number of parameters is signicantly reduced. Therefore, convolutional layers are less costly in terms of memory usage and the training time is shorter. 

Pooling Layer 

Convolutional layers are designed in such a way the spatial dimensions are preserved and the depth is increased along the network ow. However, it is practical to reduce spatial dimensions, especially in higher layers. Dimensions reduction can be obtained by using stride when convolving, leading to dilution of receptive elds overlap. Nev-ertheless, a more straightforward technique was developed called a pooling layer. An 
input is partitioned into non-overlapping rectangles and the layer simply outputs a grid of maximum values of each rectangle. In practise, pooling layers are inserted often in between convolutional layers to reduce dimensionality. 

Recurrent Neural Networks 

Convolutional and fully connected layers are designed to process input in one time step without temporal context. Nonetheless, some tasks require concerning sequences where data are temporally interdependent. For that, a Recurrent Neural Network (RNN) { an extension of fully connected layers { has been introduced. RNNs are neural networks concerning information from previous time steps . 

RNNs are used in a variety of tasks: transforming a static input into a sequence (e.g. image captioning); processing sequences into a static output (e.g. video labelling); or transforming sequences into sequences (e.g. automatic translation). 

A simple recurrent network is typically designed by taking the layer's output from the previous step and concatenating it with the current step input: 

	yt = f(xt; yt 1) 	(10)

The function f is a standard fully-connected layer that processes both inputs indis-tinctly as one vector. Due to its simplicity, this approach is rather not sucient and does not yield promising results [19]. Thus, in past years, a great number of meaningful designs have been tested [20]. The notion was advanced and designs have become more complex. For example, an inner state vector was introduced to convey information between times steps: 
	ht; yt = f(xt; ht 1) 	(11)

The most popular architecture nowadays is a Long Short-Term Memory (LSTM) [21] { a rather complex design, yet outperforming others [20]. 

Long Short-Term Memory 

A standard LSTM layer is given as follows: 


ft =  g(Wfxt + Ufht 1 + bf ) 	(12) 
it =  g(Wixt + Uiht 1 + bi) 	(13) 
ot =  g(Woxt + Uoht 1 + bo) 	(14) 
ct = ft   ct 1 + it    c(Wcxt + Ucht 1 + bc) 	(15) 
ht = ot    h(ct) 	(16) 
	


where xt is an input vector and ht 1 is an output vector. All matrices W and U and biases b are weights that together, with g which is a logistic function Eq. (7), represent a standard neural network layer. 

Thus, the forget gate vector ft, the input gate vector it and output gate vector ot are outputs of three distinct one-layer neural nets each having its output between 1 and 1. ct is a cell state vector that, as a hidden output, is propagated to the next time. 

step. ht is an output of the LSTM cell. stands for element-wise multiplication, c and h are usually set to tanh. 

Note that ct is a combination of the previous time step ct 1, element-wisely adjusted by the forget gate ft, and the output of a neural network, gated similarly by the input gate it. 

The output of LSTM ht is a function of the cell state vector, rst squashed between 0 and 1, and then adjusted by the output gate ot. 

Connected to a network, LSTM consists typically of one layer only. LSTMs are known to preserve long-term dependencies, as shown for example by Karpathy [22]. 

•	RANDOM FOREST CLASSIFIER ALGORITHM

We have used Random Forest Algorithm to predict the diseases in our project
1.	Randomly select “k” features from total “m” featureWhere k << m
2.	Among the “k” features, calculate the node “d” using the best split point.
3.	Split the node into daughter nodes using the best split.
4.	Repeat 1 to 3 steps until “l” number of nodes has been reached.
5.	Build forest by repeating steps 1 to 4 for “n” number times to create “n” number of trees.
 
•	DECISION TREE
            It is a type of supervised Machine Learning algorithm that mainly deal with classification problem. The main objective of using decision tree is to make a training model that can be used to predict the class or values of the desired value by learning elementary decision procedure surmise from existing data (training data) . In Decision Tree algorithm, we start from root of the tree to predict the class. We collate the values of the root trait with data's trait. On the basis of differentiation, we go ground with the branch parallel to that value and move to next node. In this system Decision Tree splits the symptoms as per its classification and lowers down the dataset complexity. It is most effective Machine Learning algorithm to describe Decision tree in graphical manner. It deals with huge and complicated datasets without involvement of multiple parametric structure. With the help of training datasets, decision tree model is decided and a validation dataset decides appropriate tree size to achieve the optimal final model.

•	NAÏVE BAYES
            Naive Bayes is a type of probabilistic algorithm which is based on probability theory and Bayes Theorem to calculate the probability of diseases. A Naïve Bayes algorithm has a parallel performance with decision tree and other selected classifiers. The computation cost can be brought down significantly. It is very simple to build and useful for large dataset. Naive Bayes classify the data by calculating the probability of independent variable. After the probability of each class is computed, complete transaction is assigned to high probability class. Naïve Bayes works excellent in various complex real-world problem. the benefit of using Naïve Bayes is that it needs very less amount of training dataset to evaluate the parameters necessary for classification. Bayesian rational is functional to decision maker. The portrayal for Naive Bayes is probabilities. It is based on probability theory and Bayes Theorem to forecast the class value of unexplored dataset. A record of likelihood is kept in a report for a learned Naive Bayes model .


![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/5584209b-dd9a-4562-b6a5-4085109f307f)







7.INTRODUCTION TO TESTING 
SYSTEM TEST
The purposecoftestingeis to discovercerrors. Testingeisthecprocess of tryingetodiscoverceveryconceivableefaultorcweakness in a workproduct. Iteprovidesacway to checkethefunctionalitycof components, subcassemblies, assembliescand/or aefinishedproductcIt is theeprocessofcexercisingsoftwareewiththecintentofeensuringthatctheSoftwareesystemmeetscits requirements andcuser expectations andcdoes not fail in an unacceptablecmanner. Thereearevariousctypes of test. Here eachetest type addressesca specific testingerequirement.

4.1 UnitcTesting
Unit testingcinvolves the design ofctest cases thatevalidatethatctheinternaleprogram logic is functioningcproperly, andethatprogramcinputs produce validcoutputs. All decisionebranchesandcinternalcodeeflow should becvalidated. It is theetestingofcindividual software unitscoftheeapplication. Itcisdoneeaftertheccompletion of an individualcunitbeforeeintegration. Thiscis a structuraletesting, thatcreliesoneknowledge of itscconstruction and iseinvasive. Unit testscperform basic tests atccomponentleveleand test acspecificbusinesseprocess, application, and/orcsystem configuration. Unitctests ensure that eachcunique path of a business process performscaccurately to the documentedcspecifications and containscclearlydefinedeinputs and expectedcresults.

4.2 IntegrationcTesting
Integration testscaredesignedeto test integratedcsoftware components toedetermine if theycactually run as onecprogram.  Testingeiseventcdriven and is moreeconcerned with the basiccoutcome of screens orcfields. Integration testscdemonstrate that althoughcthe components werecindividually satisfaction, as showncby successfully unit testingcthecombination of componentscis correct and consistent.Integrationctesting is specifically aimedcat   exposing thecproblems that ariseefromtheccombinationofecomponents.
4.3 FunctionaleTest
Functional testscprovide systematic demonstrationscthat functions testedeare available as specifiedebythecbusiness and technicalerequirements, systemcdocumentation, and useremanuals.
Functionalctesting is centeredeon the followingcitems:
ValideInput               :  identifiedeclasses of valideinput must becaccepted.
InvalideInput             : identifiedcclasses of invalideinput must becrejected.
Functionse                  : identifiedcfunctions must beeexercised.
Outpute	           : identified classescof application outputscmust be   exercised.
      Systems/Procedurese   : interfacingcsystems or procedures must becinvoked.
Organizationcand preparation of functionalctests is focusedeon requirements, keyefunctions, orcspecialtestecases. In addition, systematicccoveragepertainingetoidentifycBusinessprocesseflows; datacfields, predefinedeprocesses, andcsuccessiveprocessesemustbecconsideredforetesting. Beforecfunctionaltestingeiscompletecadditionaltestseareidentifiedcand the effective valueeofcurrentetestsiscdetermined.

4.4 SystemcTest
System testingcensures that theeentireintegratedcsoftwaresystememeets requirements. Itetests a configuration to ensurecknown and predictableeresults. Ancexampleofesystemtestingcistheeconfigurationorientedcsystemintegrationetest. Systemctesting is basedeon process descriptionseandcflows, emphasizingepre-drivencprocesslinkseandintegrationcpoints.

4.5 White BoxcTesting
                        White BoxcTesting is aetestingincwhich in whichethesoftwarectesterhaseknowledge of thecinnereworkings, structureeandclanguage of theesoftware, or ateleastitsepurpose. It is purpose. It iscused to testeareasthatccannotbeereachedfromcablackebox level.
4.6 BlackeBox Testing
BlackcBox Testing isetesting the softwarecwithout any knowledgeeofthecinnereworkings, structureeorclanguage of theemodulebeingctested. Blackebox tests, as mostcotherkindseof tests, must becwrittenfromeadefinitivecsourceedocument, such ascspecificationorerequirementsdocumentc such asespecification or requirementscdocument. It is aetestingincwhichtheesoftwareunderctestisetreated, as a blackcbox, youecannot “see” intoeit. Thectestprovideseinputs and respondsctooutputsewithoutconsideringchow the softwareeworks.

4.7 UniteTesting
Unitctestingiseusually conductedas parteofaecombinedcodecandunitetest phase ofethesoftwareclifecycle, althougheit is notcuncommonforecodingandcunitetesting to be conductedeastwocdistinctephases.

4.8 Test StrategycandApproache
Features  testingcwillbeeperformedmanuallycand functional tests will becwritten in detail.
Testcobjectives
•	All field entries mustcwork properly.
•	Pages mustcbe activated from the identifiedclink.
•	The entrycscreen, messages and responsescmust not be delayed.

Features tocbe tested
•	Verify that the entriescare of the correctcformat
•	No duplicatecentries should becallowed
•	All linkscshould take the user to the correctcpage.

Integration Testing
Softwarecintegration testing is thecincremental integration testing ofctwo or more integrated softwareccomponents on a single platformcto produce failures caused bycinterface defects.
The taskcof the integration testcis to check that componentscor software applications, e.g. components in acsoftware system or – onecstep up – software applicationscat the company level – interactwithoutcerror.
TestcResults:All the test casescmentioned above passedcsuccessfully. No defects encountered.
AcceptancecTesting
User Acceptance Testingcis a critical phase of anycproject and requires significantcparticipation by the endcuser. It also ensures thatcthe system meets the functionalcrequirements.
TesteResults:Allcthetestecasesmentionedcabovepassedesuccessfully. Nocdefects encountered.







8.CONCLUSION

We have compared the performance of two machine learning methods ( kNN and Random Forests) used for predicting traffic flow. The results show that simple naïve methods, such as historical average, are surprisingly effective when making long-term predictions (more than one hour into the future), while using current traffic measurement -ts as naïve method for prediction works well when making more short-term predictions (less than 1h). This is to be expected, since current traffic situation effect more on traffic in the nearby future, then on traffic in a few hours or days. By using less complex models, optimal model parameters are found more readily, the models run a lot faster, and they are easier to understand and maintain. We also state that the main disadvantage of models presented in this research, is its inability to predict unusual traffic events. Even though common traffic status is informative for a commuter in a new environment, unusual traffic is the most informative information for local commuter who is aware of usual traffic. The main reason for this disadvantage is that current models uses only historical traffic data. Since, some of unusually traffic events are caused by other related events (such as nearby traffic accidents, bad weather, holidays, etc.), we believe that by including additional data sources in the model, prediction of such events could be significantly improved. Therefore, our future plan is to collect several quality traffic related data sources (such as traffic alerts, special days statuses, bigger social events, etc.) and fuse them with loop counters data in order to generate better traffic prediction models.
SNAPSHOTS 

![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/29dd9d0b-4fa3-498a-ba57-50b272a1d971)

![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/e78cf698-c37f-4be9-ab91-f7ba98197996)

![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/2895757f-4ff6-41e1-b5fe-b33d417761d4)

![image](https://github.com/SakshiNegi410/Traffic-flow-prediction/assets/125671478/c914f7a8-eaef-4cbb-a1c2-fd6742888d01)
















 




 




 






 




 












