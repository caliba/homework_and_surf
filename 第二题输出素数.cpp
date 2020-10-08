#include <iostream>
using namespace std;
#define Length 2000000 
int contain[Length]={0};
int main()
{
	int number;
	int count=0; 
	int num=1;
	cin>>number;
	//初始化所有数序列 
	for (int i=0;i<number-1;i++) 
	{
		contain[i]=i+2;
	}
	for (int i=1;i<number-1;i++) 
	{
        for (int j=1;j<=i;j++) 
		{
            if (contain[j-1]==0)
            {
            	continue;
            }
            else 
			{
                if (contain[i]%contain[j-1]==0) 
				{
                    contain[i]=0;
                    break;
                }
            }
        }
    }
    
    for (int i=0;i<=number;i++) 
	{
        if (contain[i]!= 0) 
		{
            count++;
        }
    }
    
    
	for (int i=0;i<=number;i++) 
	{
        if ((contain[i]!= 0)&&(count!=num)) 
		{
			num++;
            cout <<contain[i] << " ";
        }
        else if((contain[i]!= 0)&&(count==num))
        {
        	cout <<contain[i];
        }
    }
	
	
	
	return 0;
}
