### A Pluto.jl notebook ###
# v0.19.2

using Markdown
using InteractiveUtils

# ╔═╡ 94053080-c5c5-11ec-2e39-638d9d7701a9
begin
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ 9f396ce9-295b-4d80-8020-542d859abfda
begin
	using BenchmarkTools
	using CUDA
	using PlutoUI
	using Plots
	using StaticArrays

	import LinearAlgebra: norm

	AA = AbstractArray
	AF = AbstractFloat
end;

# ╔═╡ 216c518e-a625-4c1a-9164-a8e47ae56e89
md"""
We will either operate in the meter-kilogram-seconds regime or in the AU-M⊙-day regime. The latter will is favoured for the nicer numbers.

The values of the constants are taken from CODATA/NIST when available.
"""

# ╔═╡ 63afcba6-eb71-4441-98b2-2be887c6fd27
begin
	const MKS = false

	if MKS
		# Standard G in m-kg-s
		const NEWTON_G = 6.674_30e-11
	else
		# Normalized to AU-M⊙-day
		const NEWTON_G = 2.959_122e-4	
	end
	
	const METER_PER_AU = 1.495_978_706_91e11
	const KG_PER_SOLAR = 1.988_55e30
	const SEC_PER_DAY = 86_400
	const SEC_PER_YEAR = 31_557_600
end;

# ╔═╡ 0982b961-eef9-433b-806c-2d8c03a1de7c
md"""
## Define body properties
"""

# ╔═╡ bce0259d-03fc-4422-b4ff-c2ab5f513f40
struct CelestialBody{T1<:AF, T2<:AA}
	mass::T1
	position::T2
	velocity::T2

	function CelestialBody(mass::AF, position::AA, velocity::AA)
		if !MKS
			mass = mass / KG_PER_SOLAR
			position = position ./ METER_PER_AU
			velocity = velocity .* (SEC_PER_DAY/METER_PER_AU)
		end

		return new{Float64, Vector{Float64}}(mass, position, velocity)
	end
end

# ╔═╡ 87b70d89-25b6-41fb-8d1c-788910ad12d5
bodies = Dict(
	:sun => CelestialBody(KG_PER_SOLAR, [0., 0., 0.], [0., 0., 0.]),
	:mercury => CelestialBody(3.285E+23, [0., 5.7E+10, 0.], [-4.7e4, 0., 0.]),
	:venus => CelestialBody(4.867_3e24, [0., 1.082_10e11, 0.], [-3.5e4, 0., 0.]),
	:earth => CelestialBody(5.972_2e24, [0., METER_PER_AU, 0.], [-3.0e4, 0., 0.]),
	:mars => CelestialBody(2.4e24, [0., 2.2E+11, 0.], [-2.4e4, 0., 0.]),
	:ohno => CelestialBody(11 * 1E+26, [2.2E+11, -5E+11, 0.], [-1e4, 1.2e4, 0.])
)

# ╔═╡ 65f28245-a1cf-4b94-a7b2-23227fc88ad0
md"""
## Read properties and com
"""

# ╔═╡ 286ba8b0-1bf7-41ad-bac6-837f50f265c0
begin
	function transform_to_com!(x::AA, m::AA)
		x .-= sum(x .* m'; dims=2)/sum(m)
		return nothing
	end

	function read_properties(bodies::Dict{Symbol, CelestialBody{Float64, Vector{Float64}}})
		N = length(bodies)
		
		pos = zeros(3, N)
		vel = zeros(3, N)
		mass = zeros(N)

		earth_ind = 0
		
		@inbounds for (ind, (body, prop)) in enumerate(bodies)

			mass[ind] = prop.mass
			pos[:, ind] = prop.position
			vel[:, ind] = prop.velocity

			if body == :earth
				earth_ind = ind
			end
		end

		transform_to_com!(pos, mass)
		transform_to_com!(vel, mass)

		return pos, vel, mass, earth_ind
	end
	
end

# ╔═╡ d3da79cd-c73a-4276-aa3b-7df06c27e840
begin
	(position, velocity, mass, ind) = read_properties(bodies)
	const EARTH = ind
end;

# ╔═╡ e3632e83-4e69-4373-82a6-336b535c07bb
md"""
## Define integrator
"""

# ╔═╡ 7f9145ef-4585-4d85-8dee-4d01f0f8159a
function eq_of_motion!(acc::AA, rᵢ::AA, rⱼ::AA, mass::AF)
	acc .+= -NEWTON_G./norm(rᵢ .- rⱼ).^3 .* mass .* (rᵢ .- rⱼ)
end

# ╔═╡ 3f58a3b7-03a9-47fc-8179-5405b0cecded
function compute_acceleration!(acceleration::AA, position::AA, mass::AA)
	
	N = length(mass)
	for ii = 1:N
		@view(acceleration[:, ii]) .= 0
		@inbounds @simd for jj = 1:N		
			if ii ≠ jj
				eq_of_motion!(
					@view(acceleration[:, ii]),
					SVector{3, Float64}(@view(position[:, ii])),
					SVector{3, Float64}(@view(position[:, jj])),
					mass[jj]
				)
			end
		end
	end
	
end

# ╔═╡ d966399c-b70e-43cc-a2eb-fa3e55e309e3
function euler!(position::AA, velocity::AA, acceleration::AA, mass::AA, Δt::AF)
	position .+=  velocity .* Δt
    velocity .+= acceleration .* Δt
	compute_acceleration!(acceleration, position, mass)
    
	return nothing
end

# ╔═╡ fe4f4cc3-059b-4d6d-882c-bba5267c7915
function verlet!(position::AA, velocity::AA, acceleration::AA, mass::AA, Δt::AF)
    position .+= velocity .* Δt
	position .+= acceleration .* 0.5Δt^2
	
    velocity .+= acceleration .* 0.5Δt
    compute_acceleration!(acceleration, position, mass)
	velocity .+= acceleration .* 0.5Δt

    return nothing
end

# ╔═╡ d5da5f3f-86a5-4288-8c83-50b65af8ca02
function symplectic_euler!(position::AA, velocity::AA, acceleration::AA, mass::AA,  Δt::AF)
	position .+= velocity .* Δt
    compute_acceleration!(acceleration, position, mass)
    velocity .+= acceleration .* Δt

    return nothing

end

# ╔═╡ b3dfb48b-c2f6-478a-bd56-8dabfbc671c7
md"""
## Define root finder for mid ttv transit
"""

# ╔═╡ 812db9b0-1f54-408a-9545-febc08e3d492
begin
	g(pos::AF, vel::AF) = pos * vel
	δg(pos::AF, vel::AF, acc::AF) = pos * acc + vel^2
		
	function newton_ttv!(
		position::AA, 
		velocity::AA, 
		acceleration::AA, 
		mass::AA,
		integrator!::Function;
		max_iter::Int = 500, 
		tolerance::Float64 = eps(Float64)
	)
	
		Δt = 0.
		fail_tolerance = true

		@inbounds for _ = 1:max_iter
			pₓ = position[1, EARTH]
			vₓ = velocity[1, EARTH]
			aₓ = acceleration[1, EARTH]
			
			δt = -g(pₓ, vₓ)/δg(pₓ, vₓ, aₓ)
			Δt += δt
			
			integrator!(
				position, 
				velocity, 
				acceleration, 
				mass, 
				δt
			)

			if abs(position[1, EARTH] ≤ tolerance)
				position[0, EARTH] = 0.
				fail_tolerance = false
				break
			end

		end

		if fail_tolerance
			print("WARNING: Root finding tolerance not met")
		end
		
		return Δt
	end

end

# ╔═╡ 3de36c8d-c837-4c17-bc13-24402114aef0
md"""
## Simulation 🙏
"""

# ╔═╡ 02080954-c779-4b95-9475-010257e7a5db
md"""
Inputs should always be in AU-M⊙-days
"""

# ╔═╡ 95b4b2b1-2846-4940-9568-6c93807f74fb
function run_simulation!(
	position::AA, 
	velocity::AA,
	mass::AA;
	num_orbit::Int = 50,
	Δt::AF = 0.1, 
	integrator!::Function = verlet!,
)

	buffer = 25
	
	num_iter = Int(÷(
		(num_orbit + buffer) * SEC_PER_YEAR, 
		Δt * SEC_PER_DAY, 
		RoundUp
	))
	
	if MKS
		Δt *= SEC_PER_DAY
	end

	num_dims, num_bodies = size(position)
	posₕ, velₕ, accₕ = ntuple(_ -> zeros(num_dims, num_bodies, num_iter+1), 3)
	
	transit_index = zeros(num_orbit)
	transitᵢ = 1
	
	# initial conditions
	iterᵢ = 1
	posₕ[:, :, 1] = position
	velₕ[:, :, 1] = velocity
	compute_acceleration!(@view(accₕ[:, :, 1]), position, mass)

	# Setting up next index with current value overwriting in loop
	posₕ[:, :, 2] = position
	velₕ[:, :, 2] = velocity
	accₕ[:, :, 2] .= @view(accₕ[:, :, 1])
	
	while transitᵢ ≤ num_orbit
		iterᵢ += 1
		
		# update current iter values and set up next iter values for next loop
		integrator!(
			@view(posₕ[:, :, iterᵢ]),
			@view(velₕ[:, :, iterᵢ]),
			@view(accₕ[:, :, iterᵢ]),
			mass,
			Δt
		)

		# crossed mid-transit line
		if posₕ[1, EARTH, iterᵢ-1] > 0 && posₕ[1, EARTH, iterᵢ] < 0

			posₕ[:, :, iterᵢ] .= @view(posₕ[:, :, iterᵢ-1])
			velₕ[:, :, iterᵢ] .= @view(velₕ[:, :, iterᵢ-1])
			accₕ[:, :, iterᵢ] .= @view(accₕ[:, :, iterᵢ-1])
			
			ttv_Δt = newton_ttv!(
				@view(posₕ[:, :, iterᵢ]), 
				@view(velₕ[:, :, iterᵢ]), 
				@view(accₕ[:, :, iterᵢ]),
				mass,
				integrator!
			)

			transit_index[transitᵢ] = iterᵢ
			transitᵢ += 1
	
		end

		posₕ[:, :, iterᵢ+1] .= @view(posₕ[:, :, iterᵢ])
		velₕ[:, :, iterᵢ+1] .= @view(velₕ[:, :, iterᵢ])
		accₕ[:, :, iterᵢ+1] .= @view(accₕ[:, :, iterᵢ])
	end
	
	return (
		@view(posₕ[:, :, 1:iterᵢ]), 
		@view(velₕ[:, :, 1:iterᵢ]), 
		@view(accₕ[:, :, 1:iterᵢ]), 
		transit_index
	)
	
end

# ╔═╡ 2a8e8994-55c0-4cd8-ae1c-a65b475ce0c1
function run_simulation_v2!(
	position::AA, 
	velocity::AA,
	mass::AA;
	num_orbit::Int = 50,
	Δt::AF = 0.1, 
	integrator!::Function = verlet!,
	save_stride::Int = 10
)

	num_dims, num_bodies = size(position)
	history_length = ceil(Int, num_orbit * SEC_PER_YEAR/SEC_PER_DAY)
	
	# we construct a time and position history array
	# an array holding index of transits for easy extraction later as well
	tₕ = zeros(history_length+1)
	posₕ = zeros(num_dims, num_bodies, history_length+1)
	ind_transit = zeros(num_orbit)
	
	# we will use auxiliary arrays to perform the simulations
	# keeping track of previous and current values
	pos₀, vel₀, acc₀ = ntuple(_ -> zeros(num_dims, num_bodies), 3)
	pos₋₁, vel₋₁, acc₋₁ = ntuple(_ -> zeros(num_dims, num_bodies), 3)
	
	# pre-initialize 
	pos₋₁ .= position
	vel₋₁ .= velocity
	compute_acceleration!(acc₋₁, position, mass)

	pos₀ .= position
	vel₀ .= velocity
	acc₀ .= acc₋₁

	posₕ[:, :, 1] .= position

	# index and helper variables
	iterᵢ = 1 	 # loop number
	transitᵢ = 1 # ind_transit latest index
	saveᵢ = 2 	 # tₕ and posₕ latest index
	tᵢ = 0 		 # current time
	save_stride = ceil(Int, save_stride ÷ Δt)
	
	if MKS
		# integrator will need the right units!
		Δt *= SEC_PER_DAY
	end

	
	while transitᵢ ≤ num_orbit

		iterᵢ += 1
		
		# update current iter values and set up next iter values for next loop
		integrator!(
			pos₀,
			vel₀,
			acc₀,
			mass,
			Δt
		)

		# crossed mid-transit line
		if pos₀[1, EARTH] < 0 && pos₋₁[1, EARTH] > 0

			pos₀ .= pos₋₁
			vel₀ .= vel₋₁
			acc₀ .= acc₋₁
			
			ttv_Δt = newton_ttv!(
				pos₀,
				vel₀,
				acc₀,
				mass,
				integrator!
			)

			tᵢ += ttv_Δt

			tₕ[saveᵢ] = tᵢ
			posₕ[:, :, saveᵢ] .= pos₀
			ind_transit[transitᵢ] = saveᵢ
			
			transitᵢ += 1
			saveᵢ += 1
		else
			tᵢ += Δt
			
			if iterᵢ % save_stride == 0
				tₕ[saveᵢ] = tᵢ
				posₕ[:, :, saveᵢ] .= pos₀
				saveᵢ += 1
			end
		end

		pos₋₁ .= pos₀
		vel₋₁ .= vel₀ 
		acc₋₁ .= acc₀
		
	end
	
	return (
		@view(tₕ[1:saveᵢ-1]), 
		@view(posₕ[:, :, 1:saveᵢ-1]), 
		ind_transit
	)
	
end

# ╔═╡ a9392090-931e-42bb-89a7-71a9d887d912
function set_equal!(A::AA, B::AA)
	@inbounds @simd for ii ∈ 1:length(B)
		A[ii] = B[ii]
	end
end

# ╔═╡ a8aafe81-9cfd-403f-8db4-365c50b38418
function run_simulation_ttv!(
	position::AA, 
	velocity::AA,
	mass::AA;
	num_orbit::Int = 50,
	Δt::AF = 0.1, 
	integrator!::Function = verlet!,
)

	num_dims, num_bodies = size(position)
	
	# we construct a time and position history array
	# an array holding index of transits for easy extraction later as well
	tₕ = zeros(Float64, num_orbit)
	
	# we will use auxiliary arrays to perform the simulations
	# keeping track of previous and current values
	pos₀, vel₀, acc₀ = ntuple(_ -> zeros(Float64, num_dims, num_bodies), 3)
	pos₋₁, vel₋₁, acc₋₁ = ntuple(_ -> zeros(Float64, num_dims, num_bodies), 3)
	
	# pre-initialize 
	set_equal!(pos₋₁, position)
	set_equal!(vel₋₁, velocity)
	compute_acceleration!(acc₋₁, position, mass)
	
	# pos₋₁ .= position
	# vel₋₁ .= velocity
	# compute_acceleration!(acc₋₁, position, mass)

	set_equal!(pos₀, position)
	set_equal!(vel₀, velocity)
	set_equal!(acc₀, acc₋₁)


	# index and helper variables
	transitᵢ = 1 # ind_transit latest index
	tᵢ = 0 		 # current time

	if MKS
		# integrator will need the right units!
		Δt *= SEC_PER_DAY
	end

	
	while transitᵢ ≤ num_orbit
	
		# update current iter values and set up next iter values for next loop
		integrator!(
			pos₀,
			vel₀,
			acc₀,
			mass,
			Δt
		)

		# crossed mid-transit line
		if pos₀[1, EARTH] < 0 && pos₋₁[1, EARTH] > 0

			set_equal!(pos₀, pos₋₁)
			set_equal!(vel₀, vel₋₁)
			set_equal!(acc₀, acc₋₁)
			
			ttv_Δt = newton_ttv!(
				pos₀,
				vel₀,
				acc₀,
				mass,
				integrator!
			)

			tᵢ += ttv_Δt

			tₕ[transitᵢ] = tᵢ
			transitᵢ += 1
		else
			tᵢ += Δt
		end

		set_equal!(pos₋₁, pos₀)
		set_equal!(vel₋₁, vel₀)
		set_equal!(acc₋₁, acc₀)

	end
	
	return tₕ
	
end

# ╔═╡ aefb3443-6d3d-4ed9-bd7b-cc91d7bebdc8
begin
	run_simulation_ttv!(position, velocity, mass; num_orbit=1, Δt=1.);
	@time run_simulation_ttv!(position, velocity, mass; num_orbit=100, Δt=.01);
end;

# ╔═╡ de847e21-91e0-4269-8804-e6a56d94108f
begin
	run_simulation!(position, velocity, mass; num_orbit=1, Δt=1.);
	@time run_simulation!(position, velocity, mass; num_orbit=100, Δt=1. );
end;

# ╔═╡ 6c3986e5-af34-4877-893f-c502fe1fdb96
begin
	run_simulation_v2!(position, velocity, mass; num_orbit=1, Δt=1.);
	@time run_simulation_v2!(position, velocity, mass; num_orbit=100, Δt=1. );
end;

# ╔═╡ c8b632e6-b397-4359-9394-e7ec903f1ee3
# t, p, trᵢ = run_simulation_v2!(position, velocity, mass; num_orbit=50, Δt=0.01);

# ╔═╡ 39e58870-a0b6-4ae5-a66c-85be1756c2aa
# begin
# 	local ii
# 	@gif for ii in 501:50:size(p)[3]
# 		plot(
# 			@view(p[1, :, ii])', 
# 			@view(p[2, :, ii])',
# 			st=:scatter,
# 			ms=5,
# 			label = nothing
# 		)
		
# 		plot!(
# 			@view(p[1, :, ii-100:ii])', 
# 			@view(p[2, :, ii-100:ii])',
# 			label = nothing,
# 			xlabel = "AU",
# 			ylabel = "AU"
# 		)

# 		xlims!(-6, 6)
# 		ylims!(-6, 6)
# 	end
# end

# ╔═╡ 2726c764-4256-46a7-badd-07e29a5136cb
# begin
# 	a = zeros(Float64, 3, 8)
# 	b = fill(9., (3, 8))

# 	function update!(a, b)
# 		a .= @view(b[:, :])
# 	end

# 	function update2!(a, b)
# 		a .= b
# 	end

# 	function update3!(a, b)
# 		@inbounds @simd for i ∈ length(a)
# 			a[i] = b[i]
# 		end
# 	end
	
# 	update!([0, 0], [0, 0])
# 	update2!([0, 0], [0, 0])
# 	update3!([0, 0], [0, 0])
# end;

# ╔═╡ 5316d985-9bb2-4677-99d7-7e6f076debae
# @benchmark update!($a, $b)

# ╔═╡ c0fdc933-5c35-496c-be07-f4623a0a0690
# @benchmark update2!($a, $b)

# ╔═╡ a48be474-0186-4eab-b062-580b31cea91a
# @benchmark update3!($a, $b)

# ╔═╡ Cell order:
# ╠═94053080-c5c5-11ec-2e39-638d9d7701a9
# ╠═9f396ce9-295b-4d80-8020-542d859abfda
# ╟─216c518e-a625-4c1a-9164-a8e47ae56e89
# ╠═63afcba6-eb71-4441-98b2-2be887c6fd27
# ╟─0982b961-eef9-433b-806c-2d8c03a1de7c
# ╠═bce0259d-03fc-4422-b4ff-c2ab5f513f40
# ╠═87b70d89-25b6-41fb-8d1c-788910ad12d5
# ╟─65f28245-a1cf-4b94-a7b2-23227fc88ad0
# ╠═286ba8b0-1bf7-41ad-bac6-837f50f265c0
# ╠═d3da79cd-c73a-4276-aa3b-7df06c27e840
# ╟─e3632e83-4e69-4373-82a6-336b535c07bb
# ╠═7f9145ef-4585-4d85-8dee-4d01f0f8159a
# ╠═3f58a3b7-03a9-47fc-8179-5405b0cecded
# ╠═d966399c-b70e-43cc-a2eb-fa3e55e309e3
# ╠═fe4f4cc3-059b-4d6d-882c-bba5267c7915
# ╠═d5da5f3f-86a5-4288-8c83-50b65af8ca02
# ╟─b3dfb48b-c2f6-478a-bd56-8dabfbc671c7
# ╠═812db9b0-1f54-408a-9545-febc08e3d492
# ╟─3de36c8d-c837-4c17-bc13-24402114aef0
# ╟─02080954-c779-4b95-9475-010257e7a5db
# ╠═95b4b2b1-2846-4940-9568-6c93807f74fb
# ╠═2a8e8994-55c0-4cd8-ae1c-a65b475ce0c1
# ╠═a9392090-931e-42bb-89a7-71a9d887d912
# ╠═a8aafe81-9cfd-403f-8db4-365c50b38418
# ╠═aefb3443-6d3d-4ed9-bd7b-cc91d7bebdc8
# ╠═de847e21-91e0-4269-8804-e6a56d94108f
# ╠═6c3986e5-af34-4877-893f-c502fe1fdb96
# ╠═c8b632e6-b397-4359-9394-e7ec903f1ee3
# ╠═39e58870-a0b6-4ae5-a66c-85be1756c2aa
# ╠═2726c764-4256-46a7-badd-07e29a5136cb
# ╠═5316d985-9bb2-4677-99d7-7e6f076debae
# ╠═c0fdc933-5c35-496c-be07-f4623a0a0690
# ╠═a48be474-0186-4eab-b062-580b31cea91a
