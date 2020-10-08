#include<iostream>
#include<math.h>
using namespace std;
int main()
{
	int f_number;
	int b_number;
	int sum=0;
	int temp=0;
	int count=0;
	int n_a=1;
	int check_now;
	//获取输入的两个数 
	cin>>f_number>>b_number;
	if((f_number<=0)|(b_number<0)|(f_number>b_number))
	{
		cout<<"error"<<endl;
		return 0;
	} 
	for(int i=f_number;i<=b_number;i++)
	{
		sum=0;
		check_now=i;
		while(check_now!=0)
		{
			temp=check_now%10;
			sum+=pow(temp,3);
			check_now=check_now/10; 
		} 
		if(sum==i)
		{
			count++;
			//cout<<i<<endl;
		}
		
	}

	for(int i=f_number;i<=b_number;i++)
	{
		sum=0;
		check_now=i;
		while(check_now!=0)
		{
			temp=check_now%10;
			sum+=pow(temp,3);
			check_now=check_now/10; 
		} 
		if((sum==i)&&(n_a<count))
		{
			n_a++; 
			cout<<i<<endl;
		}
		else if((sum==i)&&(n_a==count))
		{
			cout<<i;
		}
		
	}

	
	
	
	if(count==0)
	{
		cout<<"no"<<endl;
	}
	
	return 0;
} 
