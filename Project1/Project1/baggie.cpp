
 
#include <windows.h> 
#include <process.h>
#include "baggie.h"

double mytimecheck(void);

// implementation file for class baggie
// v1 = N, v2= T, v3=alpha, v4= pi
baggie :: baggie(int v1_in, int v2_in, double v3_in,
						double v4_in, int pbagname)
{
	v1 = v1_in; v2 = v2_in; v3 = v3_in; v4 = v4_in;
	result = 0;
	name = pbagname;
	status = WAITING;
	iterationsdone = -1;
	resultmutex;
}

void baggie :: setconsolemutex(HANDLE consolemutexinput)
{
	consolemutex = consolemutexinput;
}
void baggie :: setmastermutex(HANDLE mastermutexinput)
{
	mastermutex = mastermutexinput;
}

void baggie :: letmein(void)
{
	char icangoin;
	int localinheavysection;
	
		icangoin = 0;
		while(icangoin == 0){
			Sleep(1000);
			WaitForSingleObject(heavysectionmutex, INFINITE);
			 
			if( (*address_of_nowinheavysection) < maxworkersinheavysection){
				/** key logic: it checks to see if the number of workers in the heavy section is less than the
				number we want to allow **/
				icangoin = 1;
				++*address_of_nowinheavysection; //** increase the count
				localinheavysection = *address_of_nowinheavysection;  
				// so localinheavysection will have the count of busy workers
			}

			ReleaseMutex(heavysectionmutex);
		}
		WaitForSingleObject(consolemutex, INFINITE);
		cout << "******worker" << name <<": I'm in and there are " << localinheavysection <<" total busy workers\n";
		// we can use localinheavysection without protecting it with a mutex, because it is a local variable to this function, i.e.
		// it is not shared with other mutexes
		ReleaseMutex(consolemutex);
}

void baggie :: seeya(void)
{
	
		WaitForSingleObject(heavysectionmutex, INFINITE);
		--*address_of_nowinheavysection;
		ReleaseMutex(heavysectionmutex); 

}


////NEED TO BE IMPLEMENTED
void baggie :: baggiecomp(void)
{
	int i, j, k, M, iteration, itsdone = 0;
	double outeriterations; 
	int othercounter;
	double value;  

	M = 1000;
	value = 0;
	iteration = 0;
	othercounter = 0;
	outeriterations = 0;
	status = RUNNING;
	for(i = 0; i < M; i++){


		letmein(); // check to see if we can become busy

		double t1 = mytimecheck();  // see the comments below.  mytimecheck() returns the time of day in milliseconds
		                            // it is defined in mytimer.cpp

		for(j = 0; j < 10*M; j++){
			for(k = 0; k < M; k++){
				value += i*v1 + j*j*v2 + k*k*k*v3 +v4;
				++iteration;
				if(iteration == 100000){
					// grab the mutex
					WaitForSingleObject(consolemutex, INFINITE);
					printf("worker %d: value %g, total iterations %g\n", name, value,
							outeriterations);
					// release the mutex
					ReleaseMutex(consolemutex);
					iteration = 0;
				}
				++othercounter;

				if(othercounter == 50000){
					char letmeout = 0;

					WaitForSingleObject(mastermutex, INFINITE);

					iterationsdone = outeriterations;

					if(status == FINISHED){ // this is a status that would be set by the master
						letmeout = 1;
					}

					ReleaseMutex(mastermutex);
					othercounter = 0;

					if(letmeout){
						//quit!!!
						WaitForSingleObject(consolemutex, INFINITE);
						printf("-->worker %d: I have been told to quit\n", name); 
						ReleaseMutex(consolemutex);
						itsdone = 1;  // variable "itsdone" is used to record the fact that we have to quit
									  // "itsdone" was initialized to 0 above
					}
				}

				if(outeriterations == 50000000){
					itsdone = 1; //quit
				}

				if(itsdone) break;
				
			}
			
			if(itsdone) break;
			++outeriterations;
		}

		double t2 = mytimecheck();  // check out to see how this function works, it's in mytimer.cpp
									// mytimecheck simply returns the time of day in milliseconds
		double tdiff;
				
		tdiff = t2 - t1;  // t1 was set above 

		WaitForSingleObject(consolemutex, INFINITE);
		printf(" >> worker %d:  I have completed heavy loop in time %g\n", name, tdiff);
		ReleaseMutex(consolemutex);

		seeya();

		WaitForSingleObject(consolemutex, INFINITE);
		printf(" >> worker %d:  I am out\n", name);
		ReleaseMutex(consolemutex);

		if(itsdone) break;
	}
	 
	result = value;
}

