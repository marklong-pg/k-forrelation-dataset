classdef KForrClass
    properties
        A % is the Hadamard matrix
        n % length of input bit strings
        k % number of functions
        posThres % positive class bound
        negThres % negative class bound
        bitStringsMat % all possible input bit strings of length n
    end
    
    methods (Access = public)
        function obj = KForrClass(n,k,posThres,negThres)
            if nargin > 0
                obj.n = n;
                obj.k = k;
                obj.A = KForrClass.getA(n);

                storeMapBitString = KForrClass.getBitStrings;
                if (isKey(storeMapBitString,n))
                    obj.bitStringsMat = storeMapBitString(n);
                else
                    obj.bitStringsMat = KForrClass.createBitStringMat(n);
                    storeMapBitString(n) = obj.bitStringsMat;
                end

                obj.posThres = posThres;
                obj.negThres = negThres;
            end
        end
        
        function [result, phi] = evaluate(obj,funcMat)
            % EVALUATE evaluate the forrelation of an ensemble of k functions
            % Inputs:
            %   funcMat : shape(2^n,k), matrix storing truth tables of all k functions
            % Ouputs:
            %   result : class value (-1,0,1 for negative, invalid, positive)
            %   phi : value of phi

            if size(funcMat,1) == 1
                funcMat = reshape(funcMat,2^obj.n,obj.k);
            end
            phi = 1 / 2^((obj.k + 1)*obj.n/2) *...
                    sum(KForrClass.recursiveOmega(funcMat,obj.A,obj.k));
            if (phi >= obj.posThres)
                result = 1;
            elseif (abs(phi) <= obj.negThres)
                result = -1;
            else
                result = 0;
            end
        end
        
        function result = evaluateOmega(obj,omega)
            % EVALUATE_OMEGA evaluate a given omega vector for classification
            theta = 1 / 2^((obj.k + 1)*obj.n/2) *...
                    sum(omega);
            if (theta >= obj.posThres)
                result = 1;
            elseif (abs(theta) <= obj.negThres)
                result = -1;
            else
                result = 0;
            end
        end

        function omega = calculateOmega(obj,funcMat)
            % CALCULATE_OMEGA calculate omega value of an ensemble of k functions
            omega = KForrClass.recursiveOmega(funcMat,obj.A,obj.k);
        end
        
        function freqVec = getSamplingFreqPolicy(obj,numSamples,algo)
            % GETSAMPLING_FREQ calculate sampling frequency of classes in the policy space
            POS = zeros(1,numSamples);
            NEG = zeros(1,numSamples); 
            INV = zeros(1,numSamples);
            parfor i = 1 : numSamples
                obj_ = obj;
                if algo == "random"
                    result = obj_.evaluate(obj_.sampleRandom(obj_.n,obj_.k,"policy",[],true));
                elseif algo == "fourier"
                    result = obj_.evaluate(obj_.sampleFourier(obj_.n,obj_.k,"policy"));
                end
                if result == 1
                    POS(i) = 1;
                elseif result == -1
                    NEG(i) = 1;
                else
                    INV(i) = 1;
                end
            end
            freqVec = (sum([POS; NEG; INV],2) / numSamples)';
        end
    end
    
    methods(Static)
        function bitStringMat = createBitStringMat(n)
            % CREATE_BIT_STRINGMAT return all possible input bit strings of length n
            bitStringMat = zeros(n,2^n);
            for i = 1:2^n
                bitStringMat(:,i) = int2bit(i-1,n);
            end
        end

        function A = createBitsProductMat(bitStringLength)
            % CREATE_BITS_PRODUCTMAT create a matrix of interaction dot products of input bit strings
            %   A(i,j) = (-1) ^ dot( binary(i-1) , binary(j-1) )
            matSize = 2^bitStringLength;
            A = zeros(matSize);
            for i = 1 : matSize
                for j = 1 : matSize
                    A(i,j) = (-1)^dot(int2bit(i-1,bitStringLength),int2bit(j-1,bitStringLength));
                end
            end
        end

        function A = getA(n)
            % GET_A get Hadamard matrix for bit string length n from cache or create and save if not exist
            storeMapA = KForrClass.getMapOfA;
            if (isKey(storeMapA,n))
                A = storeMapA(n);
            else
                fprintf("\nCreating A for n = %d ...\n", n);
                A = KForrClass.createBitsProductMat(n);
                fprintf("Done!\n")
                storeMapA(n) = A;
            end
        end

        function [v, vEncode] = target_FunVec(n,sig_bits)
            % TARGET_FUNVEC generate a function vector corresponding to given choice of sig bits
            % Inputs:
            %   n : length of input bit string
            %   sig_bits : vector containing sig_bits indices
            v = ones(2^n,1);
            vEncode = zeros(1,n);
            if ~isempty(sig_bits)
                vEncode(sig_bits) = 1;
                funcCache = KForrClass.getValueFunctionCache;
                vEncodeKey = char(convertStringsToChars(string(vEncode)))';
                if isKey(funcCache,vEncodeKey)
                    v = funcCache(vEncodeKey);
                else
                    for i = 1:2^n
                        bitString = int2bit(i-1,n);
                        product = 1;
                        for j = sig_bits
                            product = product * bitString(j);
                        end
                        v(i) = (-1)^product;
                    end
                    funcCache(vEncodeKey) = v;
                end
            end
        end
        
        function omega = recursiveOmega(funcMat,A,j)
            % RECURSIVE_OMEGA Calculate the Omega vector for a k-forrelation instance
            % Inputs:
            %   funcMat : matrix with each column as function vector (funcMat(:,i) = f_i)
            %   A : Hadamard matrix
            %   j : index of termination function
            if j == 1
                omega = funcMat(:,1);
            else
                omega = A*KForrClass.recursiveOmega(funcMat,A,j-1).*funcMat(:,j);
            end
        end

        function res = fourier_transform(X)
            % FOURIER_TRANSFORM calculate the fourier transform of a design matrix
            % Inputs:
            %   X : matrix of functions as column vectors in value form
            n = log2(size(X,1));
            A = KForrClass.getA(n);
            res = 1/2^(n/2) * A * X;
        end

        function [dataset, storage, freq] = getDatasetRandomSamp(space,n,k,posThres,negThres,class,numSamples,wVec)
            % GETDATASET_RANDOM Generate a balanced k-Forrelation dataset with Random Sampling
            % Inputs:
            %   space : "general" or "policy"
            %   n : int, length of input bitstring
            %   k : int, number of function
            %   posThres : positive threshold
            %   negThres : negative threshold
            %   class: -1, 0, or 1 for Negative, Invalid, Positive
            %   numSamples: number of samplings to draw 
            %   wVec : [double], weight vector for functions in the policy
            %           space. requires: length(wVec) = number of functions in
            %           policy space
            % Returns:
            %   dataset : shape(numSamples, n x k) each row is a k-Forrelation instance in encoded form
            %   storage : shape(numSamples, 2^n x k) each row is a k-Forrelation instance in value form
            %   freq : frequency of sampling example in the class from the space

            KForrInstance = KForrClass(n,k,posThres,negThres);
            spmd
                numSamples_each = round(numSamples/spmdSize);
                storage_ = [];
                dataset_ = [];
                KForrInstance_ = KForrInstance;
                for i = 1:numSamples_each
                    [funcMat, encode] = KForrClass.sampleRandom(n,k,space,wVec,true);
                    result = KForrInstance_.evaluate(funcMat);
                    if result == class
                        storage_ = [storage_; funcMat(:)'];
                        if space == "policy"
                            dataset_ = [dataset_; encode];
                        end
                    end
                end
            end
            storage = KForrClass.merge_composite(storage_);
            dataset = KForrClass.merge_composite(dataset_);
            freq = size(storage,1)/numSamples;
            if space == "policy"
                [storage, dataset, ~] = KForrClass.cleanUp(n,storage,dataset,[]);
            else
                storage = unique(storage,"rows","stable");
                storage(~any(storage,2),:)=[];
                dataset = [];
            end
%             fprintf('\nRand Samp: Class %d: n=%d, k=%d --- %.3f%%',class,n,k,freq*100);
        end

        function [dataset, storage, freq] = getDatasetFourier(n,k,posThres,negThres,class,numSamples)
            % GETDATASET_FOURIER Generate a balanced k-Forrelation dataset with Fourier Sampling
            % Inputs:
            %   (space is policy by default)
            %   n : int, length of input bitstring
            %   k : int, number of function
            %   posThres : positive threshold
            %   negThres : negative threshold
            %   class: -1, 0, or 1 for Negative, Invalid, Positive
            %   numSamples: number of samplings to draw 
            % Returns:
            %   dataset : shape(numSamples, n x k) each row is a k-Forrelation instance in encoded form
            %   storage : shape(numSamples, 2^n x k) each row is a k-Forrelation instance in value form
            %   freq : frequency of sampling example in the class from the space
            KForrInstance = KForrClass(n,k,posThres,negThres);
            spmd
                numSamples_each = round(numSamples/spmdSize);
                storage_ = [];
                dataset_ = [];
                KForrInstance_ = KForrInstance;
                for i = 1:numSamples_each
                    [funcMat, encode] = KForrClass.sampleFourier(n,k,"policy",true);
                    result = KForrInstance_.evaluate(funcMat);
                    if result == class
                        storage_ = [storage_; funcMat(:)'];
                        dataset_ = [dataset_; encode];
                    end
                end
            end
            storage = KForrClass.merge_composite(storage_);
            dataset = KForrClass.merge_composite(dataset_);
            freq = size(dataset,1) / numSamples;
            [storage, dataset, ~] = KForrClass.cleanUp(n,storage,dataset,[]);
%             fprintf('Vanilla Fourier: Class %d: n=%d, k=%d --- %.3f%%',class,n,k,freq*100);
        end
        
        function [funcPolicySpace, funcPolicySpaceEncode] = getPolicySpace(n)
            % GET_POLICY_SPACE obtain matrices with functions in the policy space
            SIG_BIT_LIMIT = 3;
            policySpaceTable = KForrClass.getPolicySpaceCache();
            policySpaceTableEncode = KForrClass.getPolicySpaceEncodeCache();
            if ~isKey(policySpaceTable,n)
                fprintf("\nCreating Policy Space for n = %d...\n", n);
                funcSpaceValue = [];
                funcSpaceEncode = [];
                for N_sig_bits = 0:SIG_BIT_LIMIT
                    combinations = nchoosek(1:n,N_sig_bits);
                    for row = 1:size(combinations,1)
                        [vec, vecEncode] = KForrClass.target_FunVec(n,combinations(row,:));
                        funcSpaceValue = [funcSpaceValue, vec];
                        funcSpaceEncode = [funcSpaceEncode, vecEncode'];
                    end
                end
                fprintf("Done!\n")
                policySpaceTable(n) = funcSpaceValue;
                policySpaceTableEncode(n) = funcSpaceEncode;
            end
            funcPolicySpace = policySpaceTable(n);
            funcPolicySpaceEncode = policySpaceTableEncode(n);
        end

        function [funcMat, encode] = sampleRandom(n,k,space,wVec,checkPolicyValid)
            % SAMPLERANDOM Randomly sample a k-Forrelation instance
            % Inputs:
            %   n : int, length of input bitstring
            %   k : int, number of function
            %   space : str, "general" or "policy"
            %   wVec : [double], weight vector for functions in the policy
            %           space. requires: length(wVec) = number of functions in
            %           policy space
            %   checkPolicyValid : boolean, whether to enforce validity in the policy space
            % Returns:
            %   funcMat : shape (2^n,k), stores value table of all k functions
            %   encode : shape (n,k), stores one-hot encoded form of all k functions
            if space == "policy"
                [valueSpace, encodeSpace] = KForrClass.getPolicySpace(n);
                isValid = false;
                while ~isValid
                    if ~isempty(wVec)
                        chosen_funcs = randsample(1:size(valueSpace,2),k,true,wVec);
                    else
                        chosen_funcs = randsample(size(valueSpace,2),k,true);
                    end
                    funcMat = valueSpace(:,chosen_funcs);
                    encode = encodeSpace(:,chosen_funcs);
                    encode = encode(:)';
                    if checkPolicyValid
                        isValid = KForrClass.isPolicyValid(n,k,encode);
                    else
                        isValid = true;
                    end
                end
            elseif space == "general"
                funcMat = (-1).^(double (rand(2^n,k)>= 0.5));
                encode = [];
            end
        end

        function [funcMat, encode] = sampleFourier(n,k,space,checkPolicyValid)
            % SAMPLEFOURIER Sample a k-Forrelation instance using fourier sampling
            % Inputs: 
            %   n : int, length of input bitstring
            %   k : int, number of function
            %   space: str, "general" or "policy"
            % Returns:
            %   funcMat : shape (2^n,k), stores value table of all k functions
            %   encode : shape (n,k), stores one-hot encoded form of all k functions
            funcMat = [];
            encode = [];
            if space == "general"
                funcMat = KForrClass.sampleFourierGeneral(n,k);
            elseif space == "policy"
                isValid = false;
                while ~isValid
                    [funcMat, encode] = KForrClass.sampleFourierPolicy(n,k);
                    if checkPolicyValid
                        isValid = KForrClass.isPolicyValid(n,k,encode);
                    else
                        isValid = true;
                    end
                end
            end
        end

        function [funcMat, v] = sampleFourierGeneral(n,k)
            % SAMPLE_FOURIER_GENERAL Sample a k-Forrelation instance using 
            % fourier sampling in the general space
            A = KForrClass.getA(n);
            if k == 1
                v = randn(2^n,1);
                funcMat = KForrClass.sign_(v);
            else
                [prev_funcMat, prev_v] = KForrClass.sampleFourierGeneral(n,k-1);
                f = KForrClass.sign_(KForrClass.fourier_transform(prev_v));
                funcMat = [prev_funcMat, f];
                omega = KForrClass.recursiveOmega(funcMat,A,k);
                v = KForrClass.sign_(omega) .* abs(randn(2^n,1));
            end
        end

        function [funcMat, encode, v] = sampleFourierPolicy(n,k)
            % SAMPLE_FOURIER_GENERAL Sample a k-Forrelation instance using 
            % fourier sampling in the policy space
            A = KForrClass.getA(n);
            if k == 1
                [funcMat, encode] = KForrClass.sampleRandom(n,k,"policy",[],false);
                v = KForrClass.sign_(funcMat) .* abs(randn(2^n,1));
            else
                [prev_funcMat, prev_encode, prev_v] = KForrClass.sampleFourierPolicy(n,k-1);
                [f, f_encode] = KForrClass.getClosestFromPolicy( ...
                    KForrClass.sign_(KForrClass.fourier_transform(prev_v)));
                funcMat = [prev_funcMat, f];
                encode = [prev_encode, f_encode];
                omega = KForrClass.recursiveOmega(funcMat,A,k);
                v = KForrClass.sign_(omega) .* abs(randn(2^n,1));
            end
        end

        function phi_history = getPhiDistribution(n,k,numSamples,space,samplingAlgo)
            % PHI_HISTORY get value of phi for drawn examples
            % Inputs:
            %   n : int, length of input bitstring
            %   k : int, number of function
            %   numSamples : number of samples 
            %   space : "general" or "policy"
            %   samplingAlgo : "random" or "fourier"
            KForrInstance = KForrClass(n,k,1/2,1/100);
            spmd
                numSamples_each = round(numSamples/spmdSize);
                phi_storage_ = zeros(numSamples_each,1);
                KForrInstance_ = KForrInstance;
                for i = 1:numSamples_each
                    if samplingAlgo == "random"
                        funcMat = KForrClass.sampleRandom(n,k,space,[],true);
                    elseif samplingAlgo == "fourier"
                        funcMat = KForrClass.sampleFourier(n,k,space,true);
                    end
                    [~, phi] = KForrInstance_.evaluate(funcMat);
                    phi_storage_(i) = phi;
                end
            end
            phi_history = KForrClass.merge_composite(phi_storage_);
            phi_history = phi_history(:);
        end

    
        %----Helper Functions----%

        function [func_, func_encode] = getClosestFromPolicy(func)
            % Compare a boolean func from general space to all boolean
            % funcs in policy space and randomly return a func from policy
            % space with the smallest hamming distance to the desired func
            % Returns:
            %   func_ [col] : closest function in policy space
            %   func_encode [row] : encoded form of chosen function
            n = log2(size(func,1));
            [valueSpace, encodeSpace] = KForrClass.getPolicySpace(n);
            dist = pdist2(func',valueSpace',"hamming");
            
            if size(func,2) > 1
                min_dist = min(dist,[],2);
                func_ = zeros(size(func));
                func_encode = zeros(n,size(func,2));
                parfor col = 1:size(func,2)
                    valueSpace_ = valueSpace;
                    encodeSpace_ = encodeSpace;
                    min_indices = find(dist(col,:) == min_dist(col));
                    if length(min_indices) > 1
                        chosen_func = randsample(min_indices,1);
                    else
                        chosen_func = min_indices;
                    end
                    func_(:,col) = valueSpace_(:,chosen_func);
                    func_encode(:,col) = encodeSpace_(:,chosen_func)';
                end
                func_encode = func_encode(:)';
            else
                min_indices = find(dist == min(dist));
                if length(min_indices) > 1
                    chosen_func = randsample(min_indices,1);
                else
                    chosen_func = min_indices;
                end
                func_ = valueSpace(:,chosen_func);
                func_encode = encodeSpace(:,chosen_func)';
            end
        end

        function valueFuncMat = decodeBinExample(binaryExample,n)
            % DECODE_BIN_EXAMPLE convert a binary-encoded example into a value function matrix of an
            % (n,k)-forrelation instance
            k = length(binaryExample)/n;
            valueFuncMat = zeros(2^n,k);
            for i = 1:k
                binary_func = binaryExample(n*(i-1)+1:n*i);
                valueFuncMat(:,i) = KForrClass.valueFromBinary(binary_func);
            end
        end

        function valueVec = valueFromBinary(binaryVec)
            % VALUE_FROM_BINARY get value table of a one-hot encoded function
            [valueSpace, encodeSpace] = KForrClass.getPolicySpace(length(binaryVec));
            [~,id] = ismember(binaryVec,encodeSpace',"rows");
            valueVec = valueSpace(:,id);
        end

        function [storage, dataset, to_remove] = removeInvalidInstances(n, storage, dataset)
            % REMOVE_INVALID remove instances of k-Forrelation from a dataset where there is no
            % function that depends on at least three bits
            k = size(dataset,2) / n;
            to_remove = [];
            for row = 1:size(dataset,1)
                funcMatHorizontal = reshape(dataset(row,:),n,k);
                if ~sum(sum(funcMatHorizontal,1) >= 3)
                    to_remove = [to_remove, row];
                end
            end
            storage(to_remove,:) = [];
            dataset(to_remove,:) = [];
        end

        function isValid = isPolicyValid(n,k,encodeVec)
            funcMatHorizontal = reshape(encodeVec,n,k);
            isValid = logical(sum(sum(funcMatHorizontal,1) >= 3));
        end

        function merged_data = merge_composite(c)
            % Merge composite object into one single matrix
            merged_data = [];
            for i = 1:length(c)
                merged_data = [merged_data; c{i}];
            end
        end

        function [storage, dataset, omegaCache] = cleanUp(n,storage,dataset,omegaCache)
            % CLEANUP clean a generated dataset from duplicate and invalid examples
            [dataset, uniq_ind,~] = unique(dataset,"rows","stable");
            storage = storage(uniq_ind,:);
            zeros_row = ~any(storage,2);
            dataset(zeros_row,:) = [];
            storage(zeros_row,:) = [];
            [storage, dataset, inv_rows] = KForrClass.removeInvalidInstances(n,storage,dataset);
            if ~isempty(omegaCache)
                omegaCache = omegaCache(:,uniq_ind);
                omegaCache(:,zeros_row) = [];
                omegaCache(:,inv_rows) = [];
            end
        end

        function wVecBiased = getPlusBiasedWeights(n)
            [valueSpace, ~] = KForrClass.getPolicySpace(n);
            temp = double(valueSpace > 0);
            wVecBiased = sum(temp) / sum(temp,"all");
        end

        function res = sign_(X)
            res = sign(X);
            res(res==0) = 1;
        end

        function [] = warmUp(n_vec)
            for n = n_vec
                KForrClass.getA(n);
                KForrClass.getPolicySpace(n);
            end
        end
    end

    methods(Static, Access = private)
        function map = getMapOfA()
            persistent storeMapA;
            if isempty(storeMapA)
                storeMapA = containers.Map('KeyType','double','ValueType','any');
            end
            map = storeMapA;
        end
        
        function map = getBitStrings()
            persistent storeMapBitString;
            if isempty(storeMapBitString)
                storeMapBitString = containers.Map('KeyType','double','ValueType','any');
            end
            map = storeMapBitString;
        end

        function map = getValueFunctionCache()
            persistent valueFuncCache;
            if isempty(valueFuncCache)
                valueFuncCache = containers.Map('KeyType','char','ValueType','any');
            end
            map = valueFuncCache;
        end

        function map = getPolicySpaceCache()
            persistent policySpaceCache;
            if isempty(policySpaceCache)
                policySpaceCache = containers.Map('KeyType','double','ValueType','any');
            end
            map = policySpaceCache;
        end

        function map = getPolicySpaceEncodeCache()
            persistent policySpaceEncodeCache;
            if isempty(policySpaceEncodeCache)
                policySpaceEncodeCache = containers.Map('KeyType','double','ValueType','any');
            end
            map = policySpaceEncodeCache;
        end
    end
end
   