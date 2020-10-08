#include<iostream>
using namespace std;
int main()
{
	int n1;
	int n2;
	int num1[20];
	int num2[20];

	int flag1[20]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
	int flag2[20]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
	int count=0;

	//前期输入 
	cin>>n1;
	for(int i=0;i<n1;i++)
	{
		cin>>num1[i];
	} 

	cin>>n2;
	for(int i=0;i<n2;i++)
	{
		cin>>num2[i];
	} 


	//计算第一个数据集中每个元素出现的次数 
	for(int i=0;i<n1;i++)
	{
		if(flag1[i]==0)//如果这个位置被检测过 
		{
			continue;
		}
		for(int j=i+1;j<n1;j++)
		{
			if(num1[i]==num1[j]&&flag1[j]!=0)//如果发现重复,且这个重复的没有被算
			{
				flag1[i]++;//多一个重复的
				flag1[j]=0;//找到了就让他变为0 
			} 
	
		}
	}
	for(int i=0;i<n1-1;i++)
	{
		int temp=0;
		for(int j=0;j<n1-i-1;j++)
		{
			if(num1[j]>num1[j+1])
			{
				temp=num1[j];
				num1[j]=num1[j+1];
				num1[j+1]=temp;
				temp=flag1[j];
				flag1[j]=flag1[j+1];
				flag1[j+1]=temp;
			}
		}
	}

	
	
	
	
	for(int i=0;i<n2;i++)
	{
		if(flag2[i]==0)//如果这个位置被检测过 
		{
			continue;
		}
		for(int j=i+1;j<n2;j++)
		{
			if(num2[i]==num2[j]&&flag2[j]!=0)//如果发现重复,且这个重复的没有被算
			{
				flag2[i]++;//多一个重复的
				flag2[j]=0;//找到了就让他变为0 
			} 
	
		}
	}
	//检测部分
	/*
	cout<<"第一个输入"<<endl; 
	for(int i=0;i<n1;i++)
	{
			cout<<num1[i]<<" "<<flag1[i]<<endl; 
	}  
	cout<<"第二个输入"<<endl; 
	for(int i=0;i<n2;i++)
	{
		cout<<num2[i]<<" "<<flag2[i]<<endl; 
	}  
	cout<<"测试结束"<<endl; 
	*/

	for(int i=0;i<n1;i++)
	{
		if(n1!=n2)
		{
			count=0;
			break;
		}
		for(int j=0;j<n1;j++)
		{
			if((num1[i]==num2[j])&&(flag1[i]==flag2[j]))//如果Num1中的数字和出现次数与Num2中出现的数字和次数相同 
			{
				//cout<<num1[i]<<"  "<<flag1[i]<<endl;
				flag2[j]=100;
				count++;
			}
		}
	}

	if(count==n1)
	{
		cout<<1<<endl;
	} 
	else
	{
		cout<<0<<endl;
	}
	//输出第一个数据集中每一个数字 
	

	for(int i=0;i<n1;i++)
	{
	
		if(flag1[i]!=0)
		{
			cout<<num1[i]<<" "<<flag1[i]<<endl; 
		}
	


	} 
	
	
	return 0;
}
