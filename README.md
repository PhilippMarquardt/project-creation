The application is built with extensibility in mind without breaking current capabilites. To achieve this the app is generated in a multi step approach. First a plan for the application is genreated. This is the basis for any further modifications. It will genreate a frontend, backend and database plan. It adds behaviour and descriptions to each route and component making the behaviour easily verificable and testable. Alongside it keeps track of which page uses which component which makes it very easy to gather context for the model in the future.  
The initial app can generate over 100 backend files in a single prompt with verified testing.  

---

## To get into detail:

### 1. Everything starts with the initial prompt
This is the basis for the planning model which generates the whole application plan which includes everything from backend, database, frontend, security and more.

---

### 2. This plan is then shown in the frontend
It can be modified even before generating any file. In comparison to rules that are striclty in the prompt or doing this todo style generation this will generate a very predicable outcome. You will know HOW the app will be generated before even having the app. You can directly specify the functionality that YOU want to have tested if you are not satisfied with the initial tests that are auto generated.  

#### 2.5 This plan can also be the basis for loading in any existing project
But this is for later.

---

### 3. Now the interesting part: The actual generation
There are a couple of challanges like the sheer amount of files which will never fit into the context window. To mititage that the generation is done in the following way:

- A planning model creates an order of the files to be generated (This step is nearly trivial and can potentially be done without any models since the planning model from 1 genreates a file tree structure).
- Let N be the batch size, each batch is treated as follows:
  - A model is given the current file structure and the task of finding and reading any relevant context for the generation. This is done by simple ReAct prompting.
  - It will then start and implement the N files from the batch using any tools needed (searching, creating,..).

---

### 4. After all files have been implemented
It will go into the testing loop. A test file is generated testing all endpoints.
- It will go into another round of ReAct prompting until all tests passes.

This is quite powerful as you have control over the amount of functions that are developed within a single context window. It would also work with N=1 but it would take a long time as the model would have to gather context for each single file.

---

### 5. The frontend is generated
Basically in the same loop as the above. The planning model provides the general structure, components and so on and the implementation plan will be generated and split into N batches.

---

### 6. The testing of the frontend
Of course a bit more complex than the backend. This is currently not implemtend (besides simple react component tests). Proper testing would involve selneium paried with an agent to identifiy correct beavhour planning.

---

### 7. With all the points above
You have end2end generation with testing of all functionalties.

---

## Side notes
Currently sqlite is used as database which is of course not production ready, but proves the point.  
If later on a user wants to add a feature, page, component it will always adjust the intial app description and add according beahviour. By that everything stays strcutured, right context is easily extractable and a user has full control over it.