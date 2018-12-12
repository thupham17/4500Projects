#include <Windows.h>
#include <stdio.h>
#include <stdlib.h> 
#include <process.h>

#include "baggie.h"


unsigned _stdcall comp_wrapper(void *foo);

//void comp(double, double, double, int, double *);
void comp(baggie *pbag);

int main(void)
{
	baggie *pbag1, *pbag2;
	HANDLE hThread1, hThread2;
	unsigned threadID1, threadID2;
	HANDLE consolemutex;
	 
	consolemutex = CreateMutex(NULL, 0, NULL);

	//comp(5, 6, 7, &result1);	
	hThread1 = (HANDLE)_beginthreadex(NULL, 0, &comp_wrapper, (void *)pbag1,
		0, &threadID1);

	hThread2 = (HANDLE)_beginthreadex(NULL, 0, &comp_wrapper, (void *)pbag2,
		0, &threadID2); 

	for (;;){ 
		Sleep(5000);
		WaitForSingleObject(consolemutex, INFINITE);
		printf("hello, this is main\n"); fflush(stdout);
		ReleaseMutex(consolemutex);

		double foo;
		WaitForSingleObject(pbag1->resultmutex, INFINITE);
		foo = pbag1->result;
		ReleaseMutex(pbag1->resultmutex);

		WaitForSingleObject(consolemutex, INFINITE);
		printf("bag1 result equals:  %g\n", foo);
		ReleaseMutex(consolemutex);



		WaitForSingleObject(pbag2->resultmutex, INFINITE);
		foo = pbag2->result;
		ReleaseMutex(pbag2->resultmutex);

		WaitForSingleObject(consolemutex, INFINITE);
		printf("bag2 result equals:  %g\n", foo);
		ReleaseMutex(consolemutex);

	}
BACK:
	return 0;
}

//void comp_wrapper(bag *pbag)
unsigned _stdcall comp_wrapper(void *badadd)
{
	baggie *pbag = (baggie *) badadd;

	//comp(pbag->v1, pbag->v2, pbag->v3, pbag->bagname, &(pbag->result));
	comp(pbag);

	return 0;
}

