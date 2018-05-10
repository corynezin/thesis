close
hold off
files = dir('train');
files = files(3:end);
file_names = {files.name};

for file_name = file_names
    str_name = file_name{1};
    file_str = ['train/',str_name];
    a = csvread(file_str,1,0);
    a = a(:,3);
    
    str_split = split(str_name,["_","-"]);
    hidden_size = str2num(str_split{3});
    learning_rate = str2num(str_split{4})/100;
    plot(linspace(0,50,length(a)),a,'DisplayName',sprintf('Hidden Size: %d, Learning Rate: %0.2f',hidden_size,learning_rate)); hold on
    xticks()
end
lgd = legend;
lgd.Location = 'SouthEast';

lgd.FontSize = 18;
lgd.Location = 'southeast';
xlabel('Training Time (Epochs)','FontSize',18)
ylabel('Sample Accuracy','FontSize',18)
title('Training Accuracy','FontSize',24)
% set(gca,'YScale','log')
grid on
width=800;
height=800;
ylim([0.3,1])
set(gcf,'units','points','position',[0,0,width,height])
