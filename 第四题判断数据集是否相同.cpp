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

	//ǰ������ 
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


	//�����һ�����ݼ���ÿ��Ԫ�س��ֵĴ��� 
	for(int i=0;i<n1;i++)
	{
		if(flag1[i]==0)//������λ�ñ����� 
		{
			continue;
		}
		for(int j=i+1;j<n1;j++)
		{
			if(num1[i]==num1[j]&&flag1[j]!=0)//��������ظ�,������ظ���û�б���
			{
				flag1[i]++;//��һ���ظ���
				flag1[j]=0;//�ҵ��˾�������Ϊ0 
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
		if(flag2[i]==0)//������λ�ñ����� 
		{
			continue;
		}
		for(int j=i+1;j<n2;j++)
		{
			if(num2[i]==num2[j]&&flag2[j]!=0)//��������ظ�,������ظ���û�б���
			{
				flag2[i]++;//��һ���ظ���
				flag2[j]=0;//�ҵ��˾�������Ϊ0 
			} 
	
		}
	}
	//��ⲿ��
	/*
	cout<<"��һ������"<<endl; 
	for(int i=0;i<n1;i++)
	{
			cout<<num1[i]<<" "<<flag1[i]<<endl; 
	}  
	cout<<"�ڶ�������"<<endl; 
	for(int i=0;i<n2;i++)
	{
		cout<<num2[i]<<" "<<flag2[i]<<endl; 
	}  
	cout<<"���Խ���"<<endl; 
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
			if((num1[i]==num2[j])&&(flag1[i]==flag2[j]))//���Num1�е����ֺͳ��ִ�����Num2�г��ֵ����ֺʹ�����ͬ 
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
	//�����һ�����ݼ���ÿһ������ 
	

	for(int i=0;i<n1;i++)
	{
	
		if(flag1[i]!=0)
		{
			cout<<num1[i]<<" "<<flag1[i]<<endl; 
		}
	


	} 
	
	
	return 0;
}
