classdef full_sol_reference < Y_sol
    properties
        sqr % indicate if squared version of FTT is used
        lowopt
    end


    methods
        function sol = full_sol_reference(model, sqr, poly, opt, lowopt, N, epd)
            sol@Y_sol(model, poly, opt, N, epd, 10);
            sol.lowopt = lowopt;
            sol.sqr = sqr;
            sol.SIRTs = cell(model.T, 1);
            % reinitialize for 2m + d
            sol.samples = zeros(model.d+2*model.m, N, model.T+1);
            sol.L = zeros(model.d + 2*model.m, model.d + 2*model.m, model.T);
            sol.mu = zeros(model.d + 2*model.m, model.T);
            sol.logmarginal_likelihood = 0;
        end

        function sol = solve(sol)
            sol.samples(1:sol.model.d+sol.model.m,:,1) = priorsam(sol.model, sol.N);
            for t = 1:sol.model.T
                [temp, w] = push_samples(sol.model, sol.samples(1:sol.model.d+sol.model.m, :, t), ones(1, sol.N)/sol.N, t);
                sol.samples(:,:,t+1) = [temp;...
                    sol.samples(sol.model.d+1:sol.model.d+sol.model.m, :, t)];

                fprintf("\n FTT approximation at time %d because ESS =  %.2f / %d. ", t, ESS(w), sol.N)

                [sol, fun_post] = reapprox(sol, w, t);


                % generate samples
                z = rand(sol.model.d+2*sol.model.m, sol.N);
                [r, ~] = eval_irt(sol.SIRTs{t}, z);
                sol.samples(:, :, t+1) = sol.L(:,:,t) * r + sol.mu(:, t);
                w = exp(-fun_post(r))./eval_pdf(sol.SIRTs{t}, r);
                fprintf("The total ESS is %.2f out of %d. \n", ESS(w), length(w))

                % compute ESS
                sol.ESS_all(t) = ESS(w);
                sol.weight(:, t) = w';
            end
        end

        function [sol, logfun_post] = reapprox(sol, w, t)
            tic

            stepess = ESS(w);
            sam_new = sol.samples(:,:,t+1);
            w2 = w;
            ndouble = 0;
            while stepess < sol.N/10 && t~=1
                ndouble = ndouble+1;
                sam_old2 = sol.L(:,:,t-1) * eval_irt(sol.SIRTs{t-1}, rand(sol.model.d+2*sol.model.m, 2^ndouble * sol.N)) + sol.mu(:, t-1);
                [temp, w2] = push_samples(sol.model, sam_old2(1:sol.model.d+sol.model.m, :), ones(1, 2^ndouble * sol.N), t);
                sam_new = [temp; sam_old2(sol.model.d+1:sol.model.d+sol.model.m, :)];
                stepess = sum(w2)^2/sum(w2.^2);
                if ndouble > 4
                    break
                end
            end
            fprintf('Enhanced ESS is %.2f. \n',  stepess)
            [mu_temp, L_temp, ~] = computeL(sam_new, w2);
            sam_new = datasample(sam_new', size(sam_new, 2), 'Weights', w2);
            sam_new = sam_new';
          
            L_temp = sol.epd * L_temp;
            sol.L(:, :, t) =  L_temp;
            sol.mu(:, t) = mu_temp;
            
            % give prior
            if t == 1
                logfun_prior = @(x) -log(priorpdf(sol.model, x));
            else
                sirt_temp = sol.SIRTs{t-1};
                if sirt_temp.int_dir ~= 1, sirt_temp = marginalise(sirt_temp, 1); end

                logfun_prior = @(x) -log(abs(det(sol.L(1:sol.model.d + sol.model.m, 1:sol.model.d + sol.model.m, t-1)))\eval_pdf(sirt_temp, ...
                    sol.L(1:sol.model.d + sol.model.m, 1:sol.model.d + sol.model.m, t-1)\(x - sol.mu(1:sol.model.d + sol.model.m, t-1))));
            end


            % establish fun_into_sirt
            const = 0;
            bound = sol.poly.oned_domains{1}.bound(2);
            if isa(sol.poly.oned_domains{1}, "AlgebraicMapping")
                bound = sol.poly.oned_domains{1}.scale;
            end
            z = rand(sol.model.d + 2*sol.model.m, 4e3) *2 * bound - bound;
            fun_const = @(x) fun_into_sirt(sol, logfun_prior, t, L_temp*x + mu_temp, const) - log(abs(det(L_temp)));
            const = min(fun_const(z));
            logfun_post = @(x) fun_into_sirt(sol, logfun_prior, t, L_temp*x + mu_temp, const) - log(abs(det(L_temp)));

            samples_unweighted = L_temp\(sam_new - mu_temp);
            samples_debug = samples_unweighted(:, 1:size(sam_new, 2)/2);
            samples_init = samples_unweighted(:, size(sam_new, 2)/2+1:end);
            deb = InputData(samples_init, samples_debug);


            % build FTT if needed
            if sol.sqr == 0
                % fun_post = @(x) fun_into_ftt(sol, fun_prior, t, L_temp*x + mu_temp, const);
                if t == 1
                    sirt = TTIRT(logfun_post, sol.poly, sol.opt, 'var', deb);
                else
                    sirt_temp = sol.SIRTs{t-1};
                    sirt_temp.approx.opt = sol.lowopt;
                    sirt = TTIRT(logfun_post, sirt_temp, 'var', deb);
                end
                sol.SIRTs{t} = sirt;
                % debug for FTT
            else
                % build SIRT
                if t == 1
                    sirt = TTSIRT(logfun_post, sol.poly, sol.opt, 'var', deb);
                else
                    sirt_temp = sol.SIRTs{t-1};
                    sirt_temp.approx.opt = sol.lowopt;
                    sirt = TTSIRT(logfun_post,  sirt_temp, 'var', deb);
                end
            end

            sol.logmarginal_likelihood = sol.logmarginal_likelihood + log(sirt.z) - const;
            

            sol.FTT_time = [sol.FTT_time ,toc]; 
            % disp(sol.FTT_time)
            sol.SIRTs{t} = sirt;
        end

        function logpdf = fun_into_sirt(sol, logf, t, x, const)
            logpdf = logf(x([1:sol.model.d, sol.model.d + sol.model.m+1:end], :))...
                - log(transition(sol.model, x, t)) - log(like(sol.model, x, t)) - const;
            logpdf(isnan(logpdf)) = inf;
        end


        function [thetas, sams, w, varargout] = smooth(sol, N, T)
            d = sol.model.d;
            m = sol.model.m;
            sams = zeros(m , N, T+1);
            logpdf_eall = zeros(N, T);

            sirt = sol.SIRTs{T};
            [r,f] = eval_irt(sirt, rand(d+2*m, N));
            temp2 = sol.L(:, :, T) * r + sol.mu(:, T); % recover samples using L and mu
            sams(:, :, T+1) = temp2(d+1: d+m, :); % generate smaples
            sams(:, :, T) = temp2(d+m+1: end, :); % generate smaples
            if d > 0
                thetas = temp2(1:d, :);
            else
                thetas = [];
            end

            logpdf_e = -f - log(abs(det(sol.L(:, :, T))));
            logpdf_eall(:, T) = logpdf_e;
            for t = T-2:(-1):0  % k is time, stored in k+1(second indice) in samples
                disp(t)
                sirt = sol.SIRTs{t+1};
                xt = sams(:, :, t+2);
                if d > 0
                    z = sol.L(1:d+m,1:d+m, t+1)\([thetas; xt] - sol.mu(1:d+m,t+1));
                else
                    z = sol.L(1:d+m,1:d+m, t+1)\(xt - sol.mu(1:d+m,t+1));
                end
                [r,f] = eval_cirt(sirt, z, rand(m, N));
                temp = [z ; r];
                temp2 = sol.L(:,:, t+1) * temp + sol.mu(:, t+1);
                sams(:, :, t+1) = temp2(d+m+1:end, :);
                logpdf_e = logpdf_e - f - log(abs(det(sol.L(:,:, t+1)))) + log(abs(det(sol.L(1:d+m,1:d+m, t+1))));
                logpdf_eall(:, t+1) = logpdf_e;
            end

            
            % theoretical pdf

            fulldata = zeros(d+2*m, N, T+1);
            if d > 0
                fulldata(1:d, :, :) = repmat(thetas, 1, 1, T+1);
            end
            fulldata(d+1:d+m, :, :) = sams;
            fulldata(d+m+1:end, :, 2:T+1) = sams(:,:, 1:T);

            logpdf_t = reference_logprior(sol.model, fulldata(1:d+m, :, 1));
            for k = 1:T
                logpdf_t = logpdf_t + reference_logtransition(sol.model, fulldata(:, :, k+1), k) +...
                    reference_loglike(sol.model, fulldata(:, :, k+1), k);
            end
            
            % Reference repair: retain the legacy statistic but compute
            % the finite-sample log marginal likelihood as logmeanexp.
            raw_log_weight = logpdf_t - logpdf_e;
            finite_mask = ~(isnan(raw_log_weight) | isinf(raw_log_weight));
            w_temp = raw_log_weight;
            w_temp(~finite_mask) = [];
            if isempty(w_temp)
                legacy_mean_log_weight = NaN;
            else
                legacy_mean_log_weight = mean(w_temp);
            end
            finite_fraction = sum(finite_mask(:)) / numel(finite_mask);
            if any(~finite_mask(:))
                corrected_status = 0;
                lml = NaN;
                w = zeros(size(raw_log_weight));
                importance_ess = 0;
            else
                shift = max(raw_log_weight);
                shifted = exp(raw_log_weight - shift);
                normalizer = sum(shifted);
                lml = shift + log(normalizer) - log(numel(raw_log_weight));
                w = shifted / normalizer;
                importance_ess = 1 / sum(w.^2);
                corrected_status = isfinite(lml) && isfinite(importance_ess);
                if ~corrected_status
                    w = zeros(size(raw_log_weight));
                    importance_ess = 0;
                end
            end
            if nargout >= 4
                varargout{1} = logpdf_eall;
            end
            if nargout >= 5
                varargout{2} = lml;
            end
            if nargout >= 6
                varargout{3} = struct( ...
                    'raw_log_weight', raw_log_weight, ...
                    'legacy_mean_log_weight', legacy_mean_log_weight, ...
                    'finite_fraction', finite_fraction, ...
                    'importance_ess', importance_ess, ...
                    'corrected_status', corrected_status);
            end
        end

        function ESSs = smooth_t(sol, thetas, sams, logpdf_eall, Tts)
            d = sol.model.d;
            m = sol.model.m;
            T = size(sams, 3) - 1;
            N = size(sams, 2);
            ESSs = zeros(size(Tts));

            for k = 1:length(Tts)
                t_init = Tts(k);
                logpdf_e = logpdf_eall(:, t_init)';
                % theoretical pdf

                fulldata = zeros(d+2*m, N, T-t_init+1);
                if d > 0
                    fulldata(1:d, :, :) = repmat(thetas, 1, 1, T-t_init+1);
                end
                fulldata(d+1:d+m, :, :) = sams(:, :, t_init+1:T+1);
                fulldata(d+m+1:end, :, :) = sams(:,:, t_init:T);

                logpdf_t = log(eval_pdf(sol.SIRTs{t_init}, ...
                    sol.L(:,:, t_init)\(fulldata(:, :, 1)-sol.mu(:, t_init)) ));
                for t = 1:T-t_init
                    logpdf_t = logpdf_t + log(transition(sol.model, fulldata(:, :, t+1), t_init+t)) +...
                        log(like(sol.model, fulldata(:, :, t+1), t_init+t));
                end

                w = logpdf_t - logpdf_e;
                w = exp(w - max(w));
                w = w/sum(w);
                ESSs(k) = ESS(w);
            end
        end


        function plot_sirt(sol, t)
            d = sol.model.d;
            m = sol.model.m;
            % sams = zeros(d + 2*m , N);

            sirt = sol.SIRTs{t};
            [r,~] = eval_irt(sirt, rand(d+2*m, 1e3));
            sams = sol.L(:, :, t) * r + sol.mu(:, t); % recover samples using L and mu
            
            plot_marginals(compute_marginals(sams))
        end

        function stats = plot_stats(sol, thetas, sams, w)
            % compute and plot statistics of the solution
            stats.N = size(sams, 2);
            stats.ess = ESS(w);
            disp("The sample size is")
            disp(stats.N)
            disp("The effective sample size is")
            disp(stats.ess)
            if ~isempty(thetas)
                disp("The true parameter values are")
                disp(sol.model.theta')

                stats.thetas_est = mean(thetas, 2);
                disp("The biased estimated parameter values are")
                disp(stats.thetas_est')

                stats.thetas_unbiased = sum(thetas.*w, 2);  
                disp("The unbiased estimated parameter values are (with potential large varaince)")
                disp(stats.thetas_unbiased')
            end
            
            disp("The trajactories are from unweighted biased estimation.")
            sams_est = mean(sams, 2);
            stats.sams_est = reshape(sams_est, sol.model.m, []);
            sams_temp = permute(sams, [2,1,3]);
            stats.sams25 = reshape(quantile(sams_temp, 0.025), sol.model.m, []);
            stats.sams975 = reshape(quantile(sams_temp, 0.975), sol.model.m, []);
            for k = 1:sol.model.m
                figure
                hold on
                plot(0:sol.model.T, sol.model.X(k, :));
                plot(0:sol.model.T, stats.sams_est(k, :));
                plot(0:sol.model.T, stats.sams25(k, :));
                plot(0:sol.model.T, stats.sams975(k, :));
                hold off
                legend("true", "biased mean", "0.025 quantile", "0.975 quantile")
                title("The" + k + "-th dimension of states")
            end
        end
    end
end