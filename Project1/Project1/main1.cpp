#include <windows.h> 
#include <process.h>
#include <stdio.h>
#include <stdlib.h> 
#include "baggie.h" 

/****
 This program obtains two parameters from the command-line:  N and W.  The program 
 then launches N worker threads.  Each worker will then attempt to run the computationally heavy
 section (two inner loops) of the "comp" function.   However, at most W threads are allowed
 to do so simultaneously.  This condition will be regulated with a mutex.  Let's use the terminology
 "busy" to describe a worker thread that is employed in the computationally heavy section.  Busy threads
 will either be done with the computation, or be told by the master to terminate according some iteration
 limit that is checked by the master.  As each "busy" worker terminates, any of the remaining "unbusy" 
 workers try to become "busy".  One of them will succeed.  Eventually all workers terminate and the program
 stops.
***/

unsigned _stdcall comp_wrapper(void *foo);

//void comp(double, double, double, double *); 

int main(int argc, char *argv[])
{
	FILE *in = NULL;
	char mybuffer[100];
	HANDLE *pThread;
	unsigned *pthreadID;
	HANDLE consolemutex;
	HANDLE *mastermutexes;
	int retcode = 0;
	int J, W;
	int N, T;
	double alpha, pi;
	baggie **ppbaggies;

	
	if(argc != 2){
		printf("usage: heavy.exe inputFile \n");
		retcode = 1; goto BACK;
	}
	
	in = fopen(argv[1], "r");
	if (in == NULL) {
		printf("could not open %s for reading\n", argv[1]);
		retcode = 200; goto BACK;
	}

	fscanf(in, "%s", mybuffer);
	J = atoi(mybuffer);
	fscanf(in, "%s", mybuffer);
	W = atoi(mybuffer);
	printf("J = %d, W = %d\n", J, W);

	if ((J <= 0) || (W <= 0)) {
		printf("bad value of J or W: %d %d\n", J, W);
		retcode = 1; goto BACK;
	}

	ppbaggies = (baggie **)calloc(J, sizeof(baggie *));
	/** ppbaggies is an array, each of whose members is the address of a baggie, and so the type of ppbaggies is baggie ** **/
	if (ppbaggies == NULL) {
		cout << "cannot allocate" << J << "baggies\n";
		retcode = 1; goto BACK;
	}
	pThread = (HANDLE *)calloc(J, sizeof(HANDLE));
	pthreadID = (unsigned *)calloc(J, sizeof(unsigned));
	mastermutexes = (HANDLE *)calloc(J, sizeof(HANDLE));
	if ((pThread == NULL) || (pthreadID == NULL) || (mastermutexes == NULL)) {
		cout << "cannot allocate" << J << "handles and threadids\n";
		retcode = 1; goto BACK;
	}

	for (int i = 0; i < J; i++) {
		fscanf(in, "%s", mybuffer);
		N = atoi(mybuffer);
		fscanf(in, "%s", mybuffer);
		T = atoi(mybuffer);
		fscanf(in, "%s", mybuffer);
		alpha = atof(mybuffer);
		fscanf(in, "%s", mybuffer);
		pi = atof(mybuffer);

		printf("New Baggie: N = %d, T = %d, alpha = %f, pi = %f, name=%d\n", N, T, alpha, pi, i);
		ppbaggies[i] = new baggie(N, T, alpha, pi, i);  // fake "jobs": normally we would get a list of jobs from e.g. a file
	}

	fclose(in);


	

	consolemutex = CreateMutex(NULL, 0, NULL);

	for(int i = 0; i < J; i++){
		ppbaggies[i]->setconsolemutex(consolemutex); // consolemutex shared across workers plus master
	}

	HANDLE heavymutex;
	heavymutex = CreateMutex(NULL, 0, NULL);

	int nowinheavy = 0;

	for(int i = 0; i < J; i++){
		mastermutexes[i] = CreateMutex(NULL, 0, NULL);
		ppbaggies[i]->setmastermutex(mastermutexes[i]);

		ppbaggies[i]->setmaxworkersinheavysection(W);
		ppbaggies[i]->setheavysectionmutex(heavymutex); 

		ppbaggies[i]->setnowinheavyaddress( &nowinheavy );


	}

	for(int i = 0; i < J; i++){
		pThread[i] = (HANDLE)_beginthreadex( NULL, 0, &comp_wrapper, (void *) ppbaggies[i], 
			0, 		&pthreadID[i] );
	}



	/** in this program we will have N worker threads simultaneously running; however at any time at most
	W of them will be allowed to actually be computing in a "heavy" way.  This simulates a multi-agent
	search where different search algorithms take turns.   Controlling W allows us to control how CPU time this process
	effectively is given **/


	//////NEED TO BE IMPLEMENTED
	int numberrunning = W;
	for(;numberrunning > 0;){
		Sleep(5000);
		printf("master will now check on workers\n"); fflush(stdout);

		 for(int j = 0; j < J; j++){
			double jiterations;
			char jstatus = RUNNING;

			WaitForSingleObject(mastermutexes[j], INFINITE);
			jstatus = ppbaggies[j]->getstatus();
			ReleaseMutex(mastermutexes[j]);

			if(jstatus == RUNNING){

			
				WaitForSingleObject(mastermutexes[j], INFINITE);

				jiterations = ppbaggies[j]->getmeits();

				double limit = 100000; // fake termination criterion dictated by the master
				if(jiterations > limit){
					ppbaggies[j]->setstatustofinished();					 

					printf("master: worker %d has been told to quit: its = %g!!\n", 
						j, jiterations); 
					--numberrunning;
					printf("number running : %d!!\n", numberrunning);
				} 


				ReleaseMutex(mastermutexes[j]);

				if (jiterations > 0){
					WaitForSingleObject(consolemutex, INFINITE);
					printf("master: worker %d has done %g iterations, limit: %g\n", j,
						jiterations, limit);
					ReleaseMutex(consolemutex);
				}

			}

		}
	}

	for(int j = 0; j < J; j++){
		WaitForSingleObject(pThread[j], INFINITE);
		printf("--> thread %d done\n", j); 
		delete ppbaggies[j]; // calls destructor
	}
	free(ppbaggies);
BACK:
	return retcode;
}



unsigned _stdcall comp_wrapper(void *genericaddress)
{
	baggie *pbaggie = (baggie *) genericaddress;
	pbaggie->baggiecomp();
	return 0;
}

