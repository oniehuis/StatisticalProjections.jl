using JLD2
using StatisticalProjections
using CairoMakie
using ColorSchemes
using DataFrames


function findclosest(A::AbstractVector{<:Number}, x::Number)
    length(A) ≤ 1 && return firstindex(A)
    i = searchsortedfirst(A, x)
    if i == firstindex(A)
        return i
    elseif i > lastindex(A)
        return lastindex(A)
    else
        (x - A[i-1]) < (A[i] - x) ? (return i - 1) : (return i)
    end
end


function ri_integer_bin(RIs::AbstractVector{<:Real}, IM::Matrix, ri_start::Integer, ri_stop::Integer)
    ris_new = ri_start:ri_stop
    IM_new = zeros(Float64, length(ris_new), size(IM, 2))
    for (idx, ri_new) in enumerate(ris_new)
        ri_left, ri_right = ri_new - 0.5, ri_new + 0.5
        i₀ = findclosest(RIs, ri_new)
        # If there is at least one column in the range [ri_left, ri_right]
        if ri_left ≤ RIs[i₀] < ri_right
            i = i₀
            while true
                i -= 1
                if i == 0 || RIs[i] < ri_left
                    i += 1
                    break
                end
            end
            i_left = i

            i = i₀
            while true
                i += 1
                if i > length(RIs) || ri_right ≤ RIs[i]
                    i -= 1
                    break
                end
            end
            i_right = i

            IM_new[idx, :] = mean(IM[i_left:i_right, :], dims=1)

        # If there is no column in the range [ri_left, ri_right]
        else
            throw("orphan RI $ri_new")
        end
    end
    IM_new
end


df = load("/Users/oniehuis/Desktop/PPLS-DA/Polistes_data/Polistes_data_20250427/Polistes_data_20250427.jld2")["df"]

# List all column names of data frame df:
# column_names = names(df)

##### PROCESS DATA TO PLOT TICs AND MASSSSPECTRA ######

# Get IM of all rows in which column pool_seq is true
df_pool_seq_TIC = deepcopy(df[df.pool_seq .== true, :])

# Find the smallest and the largest RI integer values represented in all runs
ri_min = ceil(Int, maximum([first(ris) for ris in df_pool_seq_TIC.RIs]))
ri_max = floor(Int, minimum([last(ris) for ris in df_pool_seq_TIC.RIs]))

ri_min = max(ri_min, 1800) # Overwrite ri_min with 1800 if it is smaller

# # Integer bin RIs and associated IM
# ris_new = Vector{UnitRange{Int64}}(undef, length(df_pool_seq_TIC.RIs))
# IM_new = Vector{Matrix{Float64}}(undef, length(df_pool_seq_TIC.RIs))

# for i in axes(df_pool_seq_TIC.RIs, 1)
#     IM = ri_integer_bin(df_pool_seq_TIC.RIs[i], df_pool_seq_TIC.IM[i], ri_min, ri_max)
#     IM_new[i] = IM
#     ris_new[i] = ri_min:ri_max
# end

# df_pool_seq_TIC.RIs = ris_new
# df_pool_seq_TIC.IM = IM_new

# # Normalize IM by dividing by the sum of IM
# @. df_pool_seq_TIC.IM /= sum(df_pool_seq_TIC.IM)

# ##### PROCESS DATA TO CONDUCT CPPLS-DA ######

# Prepare data of pool-seq samples for cppls
df_pool_seq_CPPLS = deepcopy(df[df.pool_seq .== true, :])

# Replace 0 intensities with 1
@. replace!(df_pool_seq_CPPLS.IM, 0 => 1)

# Integer bin RIs and associated IM
ris_new = Vector{UnitRange{Int64}}(undef, length(df_pool_seq_CPPLS.RIs))
IM_new = Vector{Matrix{Float64}}(undef, length(df_pool_seq_CPPLS.RIs))
for i in axes(df_pool_seq_CPPLS.RIs, 1)
    IM = ri_integer_bin(df_pool_seq_CPPLS.RIs[i], df_pool_seq_CPPLS.IM[i], ri_min, ri_max)
    IM_new[i] = IM
    ris_new[i] = ri_min:ri_max
end

df_pool_seq_CPPLS.RIs = ris_new
df_pool_seq_CPPLS.IM = IM_new


# ###### Run intensities ######

# df_pool_seq_run_intensities = deepcopy(df[df.pool_seq .== true, :])

# IM_new = Vector{Matrix{Float64}}(undef, length(df_pool_seq_run_intensities.RIs))
# for i in axes(df_pool_seq_run_intensities.RIs, 1)
#     IM = ri_integer_bin(df_pool_seq_run_intensities.RIs[i], df_pool_seq_run_intensities.IM[i], ri_min, ri_max)
#     IM_new[i] = IM
# end

# run_intensities = sum.(IM_new)
# run_intensities ./= maximum(run_intensities)

############################

# Normalize IM by dividing by the sum of IM
@. df_pool_seq_CPPLS.IM /= sum(df_pool_seq_CPPLS.IM)

# CLR transform IM
@. clr_transform!(df_pool_seq_CPPLS.IM)

# Creat one-hot encoding of BIN
Y, labels = labels_to_one_hot(df_pool_seq_CPPLS.BIN)

X = Matrix{Float64}(undef, length(df_pool_seq_CPPLS.IM), size(df_pool_seq_CPPLS.IM[1], 1) * size(df_pool_seq_CPPLS.IM[1], 2))
for i in axes(df_pool_seq_CPPLS.IM, 1)
    X[i, :] = vec(df_pool_seq_CPPLS.IM[i])
end

##### CONDUCT CPPLS-DA ######

gamma = (0.8, 1.0)
n_components = 2
println("Fitting cppls for poolseq")
@time  cppls = fit_cppls(X, Y, n_components, gamma=gamma)
println("Fitting cppls for poolseq – done")
component1 = cppls.X_scores[:, 1]
component2 = cppls.X_scores[:, 2]

nmz = length(first(df_pool_seq_CPPLS.MZs))
nri = length(ri_min:ri_max)
component = 1
loading_weights_2D = reshape(cppls.X_loading_weights[:, component], nri, nmz)
loading_weights_1D = vec(sum(loading_weights_2D, dims=2))

# ###### GENERATE FIGURE 1 ######

# CairoMakie.activate!()

# fig1 = Figure(; size=(1000, 600))

# ax1 = Axis(fig1[1, 1], xticks = 2000:500:4000, xminorticks = 1800:100:4000, ylabel="Relative intensity", xminorticksvisible = true)
# ax2 = Axis(fig1[3, 1], xticks = 2000:500:4000, xminorticks = 1800:100:4000, xlabel="Kovats retention index", ylabel="Weights", xminorticksvisible = true)
# ax3 = Axis(fig1[2, 1], xticks = 2000:500:4000, xminorticks = 1800:100:4000, ylabel="Relative intensity", xminorticksvisible = true)

# rowsize!(fig1.layout, 3, Relative(1/5))

# hidexdecorations!(ax1; label = false, ticklabels = true, ticks = false, grid = true,
#     minorgrid = false, minorticks = false)
# hidexdecorations!(ax3; label = false, ticklabels = true, ticks = false, grid = true,
#     minorgrid = false, minorticks = false)
# hidexdecorations!(ax2; label = false, ticklabels = false, ticks = false, grid = true,
#     minorgrid = false, minorticks = false)

# hidespines!(ax1, :b, :t, :r)
# hidespines!(ax2, :t, :r)
# hidespines!(ax3, :b, :t, :r)
    
# xlims!(ax1, 1800, 3799)
# xlims!(ax2, 1800, 3799)
# xlims!(ax3, 1800, 3799)

# ymax = maximum(maximum.((sum.(df_pool_seq_TIC.IM, dims=2))))
# ylims!(ax1, 0, ymax * 1.05)
# ylims!(ax3, 0, ymax * 1.05)

# # Plot TICs
# for i in axes(df_pool_seq_TIC.IM, 1)
#     if df_pool_seq_TIC.BIN[i] == "AAA9495"
#         lines!(ax1, df_pool_seq_TIC.RIs[i], vec(sum(df_pool_seq_TIC.IM[i], dims=2)), color=ColorSchemes.seaborn_colorblind6[3], alpha=1/54)
#     else
#         lines!(ax3, df_pool_seq_TIC.RIs[i], vec(sum(df_pool_seq_TIC.IM[i], dims=2)), color=ColorSchemes.seaborn_colorblind6[1], alpha=1/54)
#     end
# end

# y_A = [weight > 0 ? weight : 0 for weight in loading_weights_1D]
# y_B = [weight < 0 ? weight : 0 for weight in loading_weights_1D]

# # Plot loading weights
# barplot!(ax2, collect(ri_min:ri_max), y_A, gap=0, color=ColorSchemes.seaborn_colorblind6[3], strokewidth=0)
# barplot!(ax2, collect(ri_min:ri_max), y_B, gap=0, color=ColorSchemes.seaborn_colorblind6[1], strokewidth=0)

# #ylims!(ax2, -0.05, 0.05)

# display(fig1)
# save("/Users/oniehuis/Desktop/figure1.svg", fig1)

###### GENERATE FIGURE 2 ######

function color_vector(Y)
    color = []
    for i in axes(Y, 1)
        #if i == 23 || i == 214
        #    push!(color, :green)
        if Y[i][1] == 1
            push!(color, ColorSchemes.seaborn_colorblind6[3])
        else
            push!(color, ColorSchemes.seaborn_colorblind6[1])
        end
    end
    color
end

function color_vector(Y, run_intensities)
    color = []
    for i in axes(Y, 1)
        #if i == 23 || i == 214
        #    push!(color, :green)
        if Y[i][1] == 1
            push!(color, (ColorSchemes.seaborn_colorblind6[3], run_intensities[i]))
        else
            push!(color, (ColorSchemes.seaborn_colorblind6[1], run_intensities[i]))
        end
    end
    color
end

fig2 = Figure() #; size=(1000, 600))
ax1 = Axis(fig2[1, 1], xlabel="Component 1", ylabel="Component 2")
scatter!(ax1, component1, component2, color=color_vector(Y), markersize=10)
display(fig2)

for i in axes(Y, 1)
    i_max = 0
    max_val = 0
    if Y[i][1] == 1
        component1[i] > max_val

    end

end

save("/Users/oniehuis/Desktop/Polistes_paper/Chemical_analysis/score_plot_females_PoolSeq_new.svg", fig2)


# using GLMakie

# GLMakie.activate!()

# scores = cppls.X_scores

# positions = Observable([Point2f(convert(Float32, scores[i, 1]), convert(Float32, scores[i, 2])) for i in axes(scores, 1)])

# # fig4, ax4, p = scatter(positions)

# fig4 = Figure(; size=(1000, 600))
# ax4 = Axis(fig4[1, 1], title="First Two Components from cppls_fit", xlabel="Component 1", ylabel="Component 2")
# p = scatter!(ax4, positions, color=color_vector(Y), markersize=10)

# on(events(fig4).mousebutton, priority = 2) do event
#     if event.button == Mouse.left && event.action == Mouse.press
#             plt, i = pick(fig4)
#             if plt == p
#                 println(i, " ", df_pool_seq_CPPLS.sample_tube[i])
#             end
#     end
#     return Consume(false)
# end

# fig4

####### Project all females into the cppls space ########

# Prepare data of pool-seq samples for cppls
df_non_pool_seq = deepcopy(df[df.pool_seq .== false, :])
df_non_pool_seq_F = deepcopy(df_non_pool_seq[df_non_pool_seq.sex .== "F", :])

# Replace 0 intensities with 1
@. replace!(df_non_pool_seq_F.IM, 0 => 1)

# Integer bin RIs and associated IM
ris_new = Vector{UnitRange{Int64}}(undef, length(df_non_pool_seq_F.RIs))
IM_new = Vector{Matrix{Float64}}(undef, length(df_non_pool_seq_F.RIs))
for i in axes(df_non_pool_seq_F.RIs, 1)
    IM = ri_integer_bin(df_non_pool_seq_F.RIs[i], df_non_pool_seq_F.IM[i], ri_min, ri_max)
    IM_new[i] = IM
    ris_new[i] = ri_min:ri_max
end

df_non_pool_seq_F.RIs = ris_new
df_non_pool_seq_F.IM = IM_new

# Normalize IM by dividing by the sum of IM
@. df_non_pool_seq_F.IM /= sum(df_non_pool_seq_F.IM)

# CLR transform IM
@. clr_transform!(df_non_pool_seq_F.IM)

# Creat one-hot encoding of BIN
Y, labels = labels_to_one_hot(df_non_pool_seq_F.BIN)

X = Matrix{Float64}(undef, length(df_non_pool_seq_F.IM), size(df_non_pool_seq_F.IM[1], 1) * size(df_non_pool_seq_F.IM[1], 2))
for i in axes(df_non_pool_seq_F.IM, 1)
    X[i, :] = vec(df_non_pool_seq_F.IM[i])
end

scores = calculate_scores(cppls, X)

fig3 = Figure() # ; size=(1000, 600))
ax1 = Axis(fig3[1, 1], xlabel="Component 1", ylabel="Component 2")
scatter!(ax1, scores[:, 1], scores[:, 2], color=color_vector(Y), markersize=10)
display(fig3)

save("/Users/oniehuis/Desktop/Polistes_paper/Chemical_analysis/score_plot_projection_females_new.svg", fig3)

# # ###### Interactive plot #######

# # using GLMakie

# # GLMakie.activate!()

# # positions = Observable([Point2f(convert(Float32, scores[i, 1]), convert(Float32, scores[i, 2])) for i in axes(scores, 1)])

# # # fig4, ax4, p = scatter(positions)

# # fig4 = Figure(; size=(1000, 600))
# # ax4 = Axis(fig4[1, 1], title="First Two Components from cppls_fit", xlabel="Component 1", ylabel="Component 2")
# # p = scatter!(ax4, positions, color=color_vector(Y), markersize=10)

# # on(events(fig4).mousebutton, priority = 2) do event
# #     if event.button == Mouse.left && event.action == Mouse.press
# #             plt, i = pick(fig4)
# #             if plt == p
# #                 println(i, " ", df_non_pool_seq_F.sample_tube[i])
# #             end
# #     end
# #     return Consume(false)
# # end

# # fig4

# # fig = Figure(; size=(1200, 600))
# # ax2 = Axis(fig[1, 1], xlabel="Kovats RI", ylabel="Relative Intensity", title="Loading 1D")
# # display(fig)

# ##### Extraction of mass spectra #####

# # RI with most intense loading (absolute) weight
# i = argmin(loading_weights_1D) # 893
# ri_max1 = (ri_min:ri_max)[i] # RI 2692
# # i = 893, RI = 2692; loading_weights_1D[i] = -0.7015902012921166
# # index interval around i with non-zero loading weights
# i_range = 888:899
# ri_range = (ri_min:ri_max)[i_range] # 2687:2698
# loading_weights_1D_r1 = copy(loading_weights_1D)
# loading_weights_1D_r1[i_range] .= 0

# # RI with second most intense loading (absolute) weight
# i = argmin(loading_weights_1D_r1)
# ri_max2 = (ri_min:ri_max)[i]
# # i = 972, RI = 2771; loading_weights_1D_r1[i] = -0.3861572732627759
# # index interval around i with non-zero loading weights or to local minimum
# i_range = 964:979
# ri_range = (ri_min:ri_max)[i_range] # 2763:2778
# loading_weights_1D_r2 = copy(loading_weights_1D_r1)
# loading_weights_1D_r2[i_range] .= 0

# # RI with second most intense loading (absolute) weight
# i = argmax(loading_weights_1D_r2)
# ri_max3 = (ri_min:ri_max)[i]
# # i = 1547, RI = 3346; loading_weights_1D_r1[i] = 1.1336817038886713
# # index interval around i with non-zero loading weights or to local minimum
# i_range = 1545:1554
# ri_range = (ri_min:ri_max)[i_range] # 3344:3353


# i_mode = 1547
# i_start_baseline = 1545
# i_stop_baseline = 1549
# i_start_tic = 1543
# i_stop_tic = 1551
# i_range = i_start_tic:i_stop_tic

# df_pool_seq_MS = deepcopy(df[df.pool_seq .== true, :])

# # Find the smallest and the largest RI integer values represented in all runs
# ri_min = ceil(Int, maximum([first(ris) for ris in df_pool_seq_MS.RIs]))
# ri_max = floor(Int, minimum([last(ris) for ris in df_pool_seq_MS.RIs]))

# ri_min = max(ri_min, 1800) # Overwrite ri_min with 1800 if it is smaller

# # Integer bin RIs and associated IM
# ris_new = Vector{UnitRange{Int64}}(undef, length(df_pool_seq_MS.RIs))
# IM_new = Vector{Matrix{Float64}}(undef, length(df_pool_seq_MS.RIs))

# for i in axes(df_pool_seq_MS.RIs, 1)
#     IM = ri_integer_bin(df_pool_seq_MS.RIs[i], df_pool_seq_MS.IM[i], ri_min, ri_max)
#     IM_new[i] = IM
#     ris_new[i] = ri_min:ri_max
# end

# df_pool_seq_MS.RIs = ris_new
# df_pool_seq_MS.IM = IM_new


# @. df_pool_seq_MS.IM /= sum(df_pool_seq_MS.IM)

# intsA = zeros(Float64, length(i_range))
# intsB = zeros(Float64, length(i_range))
# for i in 1:108
#     result = df_pool_seq_MS.BIN[i] == "AAA9495"
#     if result
#         global intsA += sum(df_pool_seq_MS.IM[i][i_range, :], dims=2)
#     else
#         global intsB += sum(df_pool_seq_MS.IM[i][i_range, :], dims=2)
#     end
# end

# msA = zeros(Float64, length(df_pool_seq_MS.MZs[1]))
# msA_bl = zeros(Float64, length(df_pool_seq_MS.MZs[1]))
# msB = zeros(Float64, length(df_pool_seq_MS.MZs[1]))
# msB_bl = zeros(Float64, length(df_pool_seq_MS.MZs[1]))
# for i in 1:108
#     result = df_pool_seq_MS.BIN[i] == "AAA9495"
#     if result
#         global msA += sum(df_pool_seq_MS.IM[i][i_mode, :], dims=2)
#         global msA -= (sum(df_pool_seq_MS.IM[i][i_start_baseline, :], dims=2) .+ sum(df_pool_seq_MS.IM[i][i_stop_baseline, :], dims=2)) ./ 2
#     else
#         global msB += sum(df_pool_seq_MS.IM[i][i_mode, :], dims=2)
#         global msB -= (sum(df_pool_seq_MS.IM[i][i_start_baseline, :], dims=2) .+ sum(df_pool_seq_MS.IM[i][i_stop_baseline, :], dims=2)) ./ 2
#     end
# end
# component = 1
# loading_weights_2D = reshape(cppls.X_loading_weights[:, component], nri, nmz)
# loading_weights_2D_selected = loading_weights_2D[i_mode, :]
# mz_idx_of_interest = df_pool_seq_MS.MZs[1][loading_weights_2D_selected .< 0]
# mz_idx_of_non_interest = df_pool_seq_MS.MZs[1][loading_weights_2D_selected .≥ 0]

# loading_weights_2D_selected_norm = abs.(loading_weights_2D_selected[loading_weights_2D_selected .< 0])
# loading_weights_2D_selected_norm /= maximum(loading_weights_2D_selected)

# # fig = Figure(; size=(1000, 600))
# # ax1 = Axis(fig[1, 1], title="TIC", xlabel="Kovats RI", ylabel="Relative intensity")
# # lines!(ax1, collect(ri_min:ri_max)[i_range], vec(intsA) / 54, color=ColorSchemes.seaborn_colorblind6[3])
# # lines!(ax1, collect(ri_min:ri_max)[i_range], vec(intsB) / 54, color=ColorSchemes.seaborn_colorblind6[1])
# # display(fig)

# y_max_A = maximum(vec(intsA)) / 54
# y_max_B = maximum(vec(intsB)) / 54
# y_max = max(y_max_A, y_max_B)


# fig = Figure(; size=(1000, 600))

# colsize!(fig.layout, 1, Relative(1/5))
# ax1a = Axis(fig[1, 1], title="TIC", xlabel="Kovats RI", ylabel="Relative intensity")
# lines!(ax1a, collect(ri_min:ri_max)[i_range], vec(intsA) / 54, color=ColorSchemes.seaborn_colorblind6[3])
# ax2a = Axis(fig[2, 1], title="TIC", xlabel="Kovats RI", ylabel="Relative intensity")
# lines!(ax2a, collect(ri_min:ri_max)[i_range], vec(intsB) / 54, color=ColorSchemes.seaborn_colorblind6[1])

# ax1b = Axis(fig[1, 2], title="TIC", xlabel="Kovats RI", ylabel="Relative intensity")
# vlines!(ax1b, df_pool_seq_MS.MZs[1][loading_weights_2D_selected .≤ 0]; ymax = msA[loading_weights_2D_selected .≤ 0] / (maximum(msA) * 1.05), color=:gray)
# vlines!(ax1b, df_pool_seq_MS.MZs[1][loading_weights_2D_selected .> 0]; ymax = msA[loading_weights_2D_selected .> 0] / (maximum(msA) * 1.05), color=ColorSchemes.seaborn_colorblind6[3])

# ax2b = Axis(fig[2, 2], title="TIC", xlabel="Kovats RI", ylabel="Relative intensity")
# vlines!(ax2b, df_pool_seq_MS.MZs[1][loading_weights_2D_selected .≥ 0]; ymax = msB[loading_weights_2D_selected .≥ 0] / (maximum(msB) * 1.05), color=:gray)
# vlines!(ax2b, df_pool_seq_MS.MZs[1][loading_weights_2D_selected .< 0]; ymax = msB[loading_weights_2D_selected .< 0] / (maximum(msB) * 1.05), color=ColorSchemes.seaborn_colorblind6[1])

# ylims!(ax1a, 0, y_max * 1.05)
# ylims!(ax2a, 0, y_max * 1.05)
# display(fig)