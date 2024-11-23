function  [t, condition] = plotcondition(bagfoldername)
%
%   [t, condition] = plotcondition(bagfoldername)
%
%   Plot the /condition topic saved in the bag folder.  If
%   'bagfoldername' is not given or given as 'latest', use the most
%   recent bag folder.
%
%   Also return the time/condition, with each row representing one
%   sample-time.  Note the time assumes a regular 0.01 sample rate, as
%   the message does not contain a time stamp.
%

%
%   Check the arguments
%
% If no bagfile is specified, use the most recent.
if (~exist('bagfoldername') || strcmp(bagfoldername, 'latest'))
    bagfoldername = latestbagfoldername();
end

%
%   Read the data.
%
% Load the bag.
try
    % Note: renamed to ros2bagreader() in R2022b
    bag = ros2bag(bagfoldername);
catch ME
    % Check whether the bag is incomplete.
    if (strcmp(ME.identifier, 'ros:mlros2:bag:YAMLFileNotFound'))
        disp('Recording not complete...  Is the recording stopped?');
        rethrow(ME);
        
    % Otherwise, rethrow the error.
    else
        rethrow(ME);
    end
end

% Grab the bag's start time in seconds.  Go back 10ms, as the first
% message may originated one cycle earlier.
tbag = double(bag.StartTime);
if tbag > 1e14
    tbag = tbag * 1e-9;         % If nanoseconds (R2022), convert
end
t0 = tbag - 0.010;

% Grab the /condition messages.
msgs = bagmsgs(bag, '/condition');

% Pull the data from the messages.
[t, condition] = float64data(msgs, 0.01);

        
%
%   Plot.
%
% Skip if outputing data.
if (nargout)
    disp('Skipping the plot when outputing data.');
    return;
end

% Prepare the figure.
figure(4);
clf;

% Plot.
plot(t, condition, 'LineWidth', 2);
grid on;
ylabel('Condition Number');
xlabel('Time (sec)');
title(sprintf('Condition Number in ''%s''', bagfoldername), ...
      'Interpreter', 'none');

% Name the Figure and span the full 8.5x11 page.
set(gcf, 'Name',          'Condition Number');
set(gcf, 'PaperPosition', [0.25 0.25 8.00 10.50]);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function bagfoldername = latestbagfoldername()
%
%   bagfoldername = latestbagfoldername()
%
%   Return the name of the latest bag folder including a bag file.
%   Error if there are no bag folders.
%

% Get a list of bag files in subfolders of the current folder.
d = dir('*/*.db3');

% Make sure we have at least one bag file.
if (~size(d,1))
    error('Unable to find a bag folder (including a bag file)');
end

% Find the most recently modified bag file.
[~, idx] = max([d.datenum]);

% Determine the folder that holds the bag file.
[root, name, ext] = fileparts(d(idx).folder);
bagfoldername = strcat(name,ext);

% Report.
disp(['Using bag folder ''' bagfoldername '''']);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  msgs = ros2bagmsgs(bagfoldername, topicname)
%
%   msgs = ros2bagmsgs(bagfoldername, topicname)
%
%   Extract the messages of the named topic from the bag file in the
%   give folder.  The messages are returned as a struct array.  The
%   structure contains MessageType as well as the fields of the topic.
%

% Load the bag.
try
    % Note: renamed to ros2bagreader() in R2022b
    bag = ros2bag(bagfoldername);
catch
    error(['Unable to open the bag folder ''' bagfoldername '''']);
end

% Grab the messages.
msgs = bagmsgs(bag, topicname);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  msgs = bagmsgs(bag, topicname)
%
%   msgs = bagmsgs(bag, topicname)
%
%   Extract the messages of the named topic from the given ROS2 bag.
%   The messages are returned as a struct array.  The structure
%   contains MessageType as well as the fields of the topic.
%

% Isolate the specified topic.
topic = select(bag, 'Topic', topicname);
if (~topic.NumMessages)
    warning(['No messages under topic ''' topicname '''']);
end

% Convert the messages in the topic into structure array.
msgs = cell2mat(readMessages(topic));

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  [t, data] = float64data(msgs, dt)
%
%   [t, data] = float64data(msgs, dt)
%
%   Extract the data from the given Float64 messages.  This contains
%   no time stamp, so the time assumes a constant sample step of dt.
%   The return data gives a row per time sample.
%

% Double-check the type.
if (~strcmp(msgs(1).MessageType, 'std_msgs/Float64'))
    error(['Messages are not of type std_msgs/Float64']);
end

% Check the number of samples.
M = length(msgs);

% Create the time vector, assuming a constant sample step.
t = dt * (0:M-1)';

% Extract the msgs, whether the individual elements are row or column vectors.
data = reshape(horzcat(msgs.data), 1, M)';

end
