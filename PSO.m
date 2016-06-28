function stump=PSO(x, y, d, we)


%% Problem Definition
CostFunction=@ (thr,x,y,w) stumpCost(thr,x,y,we);   % Cost Function

nVar=size(x,1);                          % Number of Decision Variables
varSize=[1 nVar];            % Size of Decision Variable Matrix

varMin=1;
varMax=nVar;


%% PSO Parameters

MaxIt=50;     % Maximum Number of Iteration
nPop=5;      % Population Size (Swarm Size)


% w=1;            % Inertia Weight
% wdamp=0.99;     % Inertia Weight Damping Ratio
% c1=2;           % Personal Learning Coefficient
% c2=2;           % Global Learning Coefficient

% Constriction Coefficients
phi1=2.05;
phi2=2.05;
phi=phi1+phi2;
chi=2/(phi-2+sqrt(phi^2-4*phi));
w=chi;          % Inertia Weight
wdamp=1;        % Inertia Weight Damping Ratio
c1=chi*phi1;    % Personal Learning Coefficient
c2=chi*phi2;    % Global Learning Coefficient

% Velocity Limits
velMax=0.1*(varMax-varMin);
velMin=-velMax;


%% Initialization

empty_particle.Position=[];

empty_particle.Stump.dim=d;
empty_particle.Stump.error=inf;
empty_particle.Stump.threshold=[];
empty_particle.Stump.less=1;
empty_particle.Stump.more=-1;

empty_particle.Velocity=0;
empty_particle.Best.Position=[];
empty_particle.Best.Stump.dim=d;
empty_particle.Best.Stump.error=inf;
empty_particle.Best.Stump.threshold=[];
empty_particle.Best.Stump.less=1;
empty_particle.Best.Stump.more=-1;


particle=repmat(empty_particle,nPop,1);
GlobalBest.Stump.dim=d;
GlobalBest.Stump.error=inf;
GlobalBest.Stump.threshold=[];
GlobalBest.Stump.less=1;
GlobalBest.Stump.more=-1;

for i=1:nPop
    particle(i).Position=randi(nVar);
    particle(i).Velocity=0;
    particle(i).Stump=CostFunction(particle(i).Position,x,y,we);
    particle(i).Best.Position=particle(i).Position;
    
    particle(i).Best.Stump=particle(i).Stump;
    
    if particle(i).Best.Stump.error<GlobalBest.Stump.error
        GlobalBest=particle(i).Best;
    end
end


BestCost=zeros(MaxIt,1);


%% PSO Main Loop

for it=1:MaxIt
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity=w*particle(i).Velocity...
            +c1*rand.*(particle(i).Best.Position-particle(i).Position)...
            +c2*rand.*(GlobalBest.Position-particle(i).Position);
        
        
        % Apply Velocity Limits
        particle(i).Velocity=max(particle(i).Velocity,velMin);
        particle(i).Velocity=min(particle(i).Velocity,velMax);
        
        % Update Position
        particle(i).Position=particle(i).Position+floor(particle(i).Velocity);

        % Velocity Mirror Effect
        IsOutSide=(particle(i).Position<varMin | particle(i).Position>varMax);
        particle(i).Velocity(IsOutSide)=-particle(i).Velocity(IsOutSide);
        
        
        % Apply Position Limits
        particle(i).Position=max(particle(i).Position,varMin);
        particle(i).Position=min(particle(i).Position,varMax);
        
        % Evaluation
         particle(i).Stump=CostFunction(particle(i).Position,x,y,we);
        
        % Update Personal Best
        
        if particle(i).Stump.error<particle(i).Best.Stump.error
            particle(i).Best.Stump=particle(i).Stump;
            particle(i).Best.Position=particle(i).Position;
            
            % Update Global Best
            if particle(i).Best.Stump.error<GlobalBest.Stump.error
                GlobalBest=particle(i).Best;
            end
        end
    end
    
    BestCost(it)=GlobalBest.Stump.error;
    disp(['Iteration ' num2str(it) ':   Best Cost = ' num2str(BestCost(it))]);
    w=w*wdamp;
end

stump.dim = d;
stump.error=GlobalBest.Stump.error;
stump.threshold=x(GlobalBest.Stump.threshold);
stump.less=GlobalBest.Stump.less;
stump.more=GlobalBest.Stump.more;

end
