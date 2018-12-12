#include <iostream> 

#define RUNNING 1
#define WAITING 0
#define FINISHED 2 
 

class baggie{
public:
	baggie(int v1input, int v2input, double v3input, double v4input, int name);
	~baggie(){ printf("worker %d says goodbye\n", name); } 
  void setconsolemutex(HANDLE consolemutexinput);
  void setmastermutex(HANDLE consolemutexinput);
  void baggiecomp();
  double getmeits(void){return iterationsdone;}
  void setstatustofinished(void){status = FINISHED;}
  int getstatus(void){ return status; }
  /////
  int v1;
  int v2;
  double v3;
  double v4;
  int bagname;
  HANDLE heavysectionmutex;
  HANDLE consolemutex;
  HANDLE mastermutex;
  HANDLE resultmutex;
  double result;
  /////
  void setheavysectionmutex(HANDLE heavysectioninput){heavysectionmutex = heavysectioninput;}
  void setmaxworkersinheavysection(int maxheavy){
	  maxworkersinheavysection = maxheavy;}
  void setnowinheavyaddress(int *paddress){address_of_nowinheavysection = paddress;}
 private:

  int name;
  double iterationsdone;
  int status;
  int maxworkersinheavysection;
  int *address_of_nowinheavysection;  /** this is the address of the integer keeping track of how many workers are busy **/
  
  void letmein(void);
  void seeya(void);
};

using namespace std;

