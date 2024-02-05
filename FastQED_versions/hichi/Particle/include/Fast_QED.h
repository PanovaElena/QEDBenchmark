#pragma once
#include "Constants.h"
#include "Ensemble.h"
#include "Grid.h"
#include "AnalyticalField.h"
#include "Pusher.h"
#include "Vectors.h"

#include "compton.h"
#include "breit_wheeler.h"
#include "macros.h"

#include <cstdint>
#include <omp.h>
#include <random>
#include <algorithm>

using namespace constants;
namespace pfc
{
    template <class TGrid>  // may be AnalyticalField or any Grid type
    class Scalar_Fast_QED_v0 : public ParticlePusher
    {
    public:

        Scalar_Fast_QED_v0() : compton(), breit_wheeler()
        {
            SchwingerField = sqr(Constants<FP>::electronMass() * Constants<FP>::lightVelocity())
                * Constants<FP>::lightVelocity() / (-Constants<FP>::electronCharge() * Constants<FP>::planck());

            coeffPhoton_probability = 1.0;
            coeffPair_probability = 1.0;

            this->rand_generator.seed();

            distribution = std::uniform_real_distribution<FP>(0.0, 1.0);
            int max_threads;

#ifdef __USE_OMP__
            max_threads = omp_get_max_threads();
#else
            max_threads = 1;
#endif
            AvalanchePhotons.resize(max_threads);
            AvalancheParticles.resize(max_threads);
            afterAvalanchePhotons.resize(max_threads);
            afterAvalancheParticles.resize(max_threads);
            timeAvalanchePhotons.resize(max_threads);
            timeAvalancheParticles.resize(max_threads);
        }

        void disable_photon_emission()
        {
            this->coeffPhoton_probability = 0.0;
        }

        void enable_photon_emission()
        {
            this->coeffPhoton_probability = 1.0;
        }

        void disable_pair_production()
        {
            this->coeffPair_probability = 0.0;
        }

        void enable_pair_production()
        {
            this->coeffPair_probability = 1.0;
        }

        void processParticles(Ensemble3d* particles, TGrid* grid, FP timeStep)
        {
            int max_threads;
#ifdef __USE_OMP__
            max_threads = omp_get_max_threads();
#else
            max_threads = 1;
#endif

            for (int th = 0; th < max_threads; th++)
            {
                AvalanchePhotons[th].clear();
                AvalancheParticles[th].clear();
                afterAvalanchePhotons[th].clear();
                afterAvalancheParticles[th].clear();
            }

            if ((*particles)[Photon].size())
                HandlePhotons((*particles)[Photon], grid, timeStep);
            if ((*particles)[Electron].size())
                HandleParticles((*particles)[Electron], grid, timeStep);
            if ((*particles)[Positron].size())
                HandleParticles((*particles)[Positron], grid, timeStep);

            for (int th = 0; th < max_threads; th++)
            {
                for (int ind = 0; ind < afterAvalanchePhotons[th].size(); ind++)
                {
                    particles->addParticle(afterAvalanchePhotons[th][ind]);
                }
                for (int ind = 0; ind < afterAvalancheParticles[th].size(); ind++)
                {
                    particles->addParticle(afterAvalancheParticles[th][ind]);
                }
            }
        }

        void Boris(Particle3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / (2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void Boris(ParticleProxy3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / (2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void HandlePhotons(ParticleArray3d& particles, TGrid* grid, FP timeStep)
        {
#pragma omp parallel for
            for (int i = 0; i < particles.size(); i++)
            {
                int thread_id;
#ifdef __USE_OMP__
                thread_id = omp_get_thread_num();
#else
                thread_id = 0;
#endif
                FP3 pPos = particles[i].getPosition();
                FP3 e, b;

                e = grid->getE(pPos);
                b = grid->getB(pPos);

                AvalancheParticles[thread_id].clear();
                AvalanchePhotons[thread_id].clear();
                timeAvalancheParticles[thread_id].clear();
                timeAvalanchePhotons[thread_id].clear();
                AvalanchePhotons[thread_id].push_back(particles[i]);
                timeAvalanchePhotons[thread_id].push_back((FP)0.0);

                RunAvalanche(e, b, timeStep);

                for (int k = 0; k != AvalanchePhotons[thread_id].size(); k++)
                    if (AvalanchePhotons[thread_id][k].getGamma() != (FP)1.0)
                        afterAvalanchePhotons[thread_id].push_back(AvalanchePhotons[thread_id][k]);


                for (int k = 0; k != AvalancheParticles[thread_id].size(); k++)
                    afterAvalancheParticles[thread_id].push_back(AvalancheParticles[thread_id][k]);
            }

            particles.clear();
        }

        void HandleParticles(ParticleArray3d& particles, TGrid* grid, FP timeStep)
        {

#pragma omp parallel for
            for (int i = 0; i < particles.size(); i++)
            {
                int thread_id;
#ifdef __USE_OMP__
                thread_id = omp_get_thread_num();
#else
                thread_id = 0;
#endif
                FP3 pPos = particles[i].getPosition();
                FP3 e, b;

                e = grid->getE(pPos);
                b = grid->getB(pPos);

                AvalancheParticles[thread_id].clear();
                AvalanchePhotons[thread_id].clear();
                timeAvalancheParticles[thread_id].clear();
                timeAvalanchePhotons[thread_id].clear();
                AvalancheParticles[thread_id].push_back(particles[i]);
                timeAvalancheParticles[thread_id].push_back((FP)0.0);

                RunAvalanche(e, b, timeStep);

                for (int k = 0; k != AvalanchePhotons[thread_id].size(); k++)
                    if (AvalanchePhotons[thread_id][k].getGamma() != (FP)1.0)
                        afterAvalanchePhotons[thread_id].push_back(AvalanchePhotons[thread_id][k]);

                particles[i].setMomentum(AvalancheParticles[thread_id][0].getMomentum());
                particles[i].setPosition(AvalancheParticles[thread_id][0].getPosition());

                for (int k = 1; k != AvalancheParticles[thread_id].size(); k++)
                    afterAvalancheParticles[thread_id].push_back(AvalancheParticles[thread_id][k]);
            }
        }

        void RunAvalanche(const FP3& E, const FP3& B, double timeStep)
        {
            int thread_id;
#ifdef __USE_OMP__
            thread_id = omp_get_thread_num();
#else
            thread_id = 0;
#endif
            vector<Particle3d>& AvalancheParticles = this->AvalancheParticles[thread_id];
            vector<Particle3d>& AvalanchePhotons = this->AvalanchePhotons[thread_id];

            vector<FP>& timeAvalancheParticles = this->timeAvalancheParticles[thread_id];
            vector<FP>& timeAvalanchePhotons = this->timeAvalanchePhotons[thread_id];

            int countParticles = 0;
            int countPhotons = 0;

            while (countParticles != AvalancheParticles.size()
                || countPhotons != AvalanchePhotons.size())
            {
                for (int k = countParticles; k != AvalancheParticles.size(); k++)
                {
                    oneParticleStep(AvalancheParticles[k], E, B, timeAvalancheParticles[k], timeStep);
                    countParticles++;
                }
                for (int k = countPhotons; k != AvalanchePhotons.size(); k++)
                {
                    onePhotonStep(AvalanchePhotons[k], E, B, timeAvalanchePhotons[k], timeStep);
                    countPhotons++;
                }
            }
        }

        void oneParticleStep(Particle3d& particle, const FP3& E, const FP3& B, FP& time, double timeStep)
        {
            int thread_id;
#ifdef __USE_OMP__
            thread_id = omp_get_thread_num();
#else
            thread_id = 0;
#endif
            while (time < timeStep)
            {
                FP3 v = particle.getVelocity();
                FP H_eff = sqr(E + (1 / Constants<FP>::lightVelocity()) * VP(v, B))
                    - sqr(SP(E, v) / Constants<FP>::lightVelocity());
                if (H_eff < 0) H_eff = 0;
                H_eff = sqrt(H_eff);
                FP gamma = particle.getGamma();
                FP chi = gamma * H_eff / SchwingerField;
                FP rate = 0.0, dt = 2 * timeStep;
                if (chi > 0.0 && coeffPhoton_probability != 0.0)
                {
                    rate = compton.rate(chi);
                    dt = getDtParticle(particle, rate, chi);
                }

                if (dt + time > timeStep)
                {
                    Boris(particle, E, B, timeStep - time);
                    time = timeStep;
                }
                else
                {
                    Boris(particle, E, B, dt);
                    time += dt;
                    FP3 v = particle.getVelocity();
                    FP H_eff = sqr(E + (1 / Constants<FP>::lightVelocity()) * VP(v, B))
                        - sqr(SP(E, v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    FP gamma = particle.getGamma();
                    FP chi_new = gamma * H_eff / SchwingerField;

                    FP delta = Photon_Generator((chi + chi_new) / (FP)2.0);

                    Particle3d NewParticle;
                    NewParticle.setType(Photon);
                    NewParticle.setWeight(particle.getWeight());
                    NewParticle.setPosition(particle.getPosition());
                    NewParticle.setMomentum(delta * particle.getMomentum());

                    this->AvalanchePhotons[thread_id].push_back(NewParticle);
                    this->timeAvalanchePhotons[thread_id].push_back(time);
                    particle.setMomentum((1 - delta) * particle.getMomentum());
                }
            }
        }

        void onePhotonStep(Particle3d& particle, const FP3& E, const FP3& B, FP& time, double timeStep)
        {
            int thread_id;
#ifdef __USE_OMP__
            thread_id = omp_get_thread_num();
#else
            thread_id = 0;
#endif

            FP3 k_ = particle.getVelocity();
            k_ = (1 / k_.norm()) * k_; // normalized wave vector
            FP H_eff = sqrt(sqr(E + VP(k_, B)) - sqr(SP(E, k_)));
            FP gamma = particle.getMomentum().norm()
                / (Constants<FP>::electronMass() * Constants<FP>::lightVelocity());
            FP chi = gamma * H_eff / SchwingerField;

            FP rate = 0.0, dt = 2 * timeStep;
            if (chi > 0.0 && coeffPair_probability != 0.0)
            {
                rate = breit_wheeler.rate(chi);
                dt = getDtPhoton(particle, rate, chi, gamma);

            }

            if (dt + time > timeStep)
            {
                particle.setPosition(particle.getPosition()
                    + (timeStep - time) * Constants<FP>::lightVelocity() * k_);
                time = timeStep;
            }
            else
            {
                particle.setPosition(particle.getPosition()
                    + dt * Constants<FP>::lightVelocity() * k_);
                time += dt;
                FP delta = Pair_Generator(chi);

                Particle3d NewParticle;
                NewParticle.setType(Electron);
                NewParticle.setWeight(particle.getWeight());
                NewParticle.setPosition(particle.getPosition());
                NewParticle.setMomentum(delta * particle.getMomentum());

                this->AvalancheParticles[thread_id].push_back(NewParticle);
                this->timeAvalancheParticles[thread_id].push_back(time);

                NewParticle.setType(Positron);
                NewParticle.setMomentum((1 - delta) * particle.getMomentum());
                this->AvalancheParticles[thread_id].push_back(NewParticle);
                this->timeAvalancheParticles[thread_id].push_back(time);

                particle.setP((FP)0.0 * particle.getP());
                time = timeStep;
            }
        }

        FP getDtParticle(Particle3d& particle, FP rate, FP chi)
        {
            FP r = -log(random_number_omp());
            r *= (particle.getGamma()) / (chi);
            return r / rate;
        }

        FP Photon_Generator(FP chi)
        {
            FP r = random_number_omp();
            return compton.inv_cdf(r, chi);
        }


        FP getDtPhoton(Particle3d& particle, FP rate, FP chi, FP gamma)
        {
            FP r = -log(random_number_omp());
            r *= (gamma) / (chi);
            return r / rate;
        }

        FP Pair_Generator(FP chi)
        {
            FP r = random_number_omp();
            if (r < 0.5)
                return breit_wheeler.inv_cdf(r, chi);
            else
                return 1.0 - breit_wheeler.inv_cdf(1.0 - r, chi);
        }

        void operator()(ParticleProxy3d* particle, ValueField field, FP timeStep)
        {}

        void operator()(Particle3d* particle, ValueField field, FP timeStep)
        {
            ParticleProxy3d particleProxy(*particle);
            this->operator()(&particleProxy, field, timeStep);
        }


        Compton compton;
        Breit_wheeler breit_wheeler;

    private:

        FP random_number_omp()
        {
            FP rand_n;
#pragma omp critical
            rand_n = distribution(rand_generator);
            return rand_n;
        }

        std::default_random_engine rand_generator;
        std::uniform_real_distribution<FP> distribution;

        vector<vector<FP>> timeAvalanchePhotons, timeAvalancheParticles;
        vector<vector<Particle3d>> AvalanchePhotons, AvalancheParticles;
        vector<vector<Particle3d>> afterAvalanchePhotons, afterAvalancheParticles;

        FP SchwingerField;

        FP coeffPhoton_probability, coeffPair_probability;
    };

    typedef Scalar_Fast_QED_v0<YeeGrid> Scalar_Fast_QED_Yee_v0;
    typedef Scalar_Fast_QED_v0<PSTDGrid> Scalar_Fast_QED_PSTD_v0;
    typedef Scalar_Fast_QED_v0<PSATDGrid> Scalar_Fast_QED_PSATD_v0;
    typedef Scalar_Fast_QED_v0<AnalyticalField> Scalar_Fast_QED_Analytical_v0;


    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////


    template <class TGrid>  // may be AnalyticalField or any Grid type
    class Scalar_Fast_QED_v1 : public ParticlePusher
    {
    public:

        Scalar_Fast_QED_v1() : compton(), breit_wheeler()
        {
            SchwingerField = sqr(Constants<FP>::electronMass() * Constants<FP>::lightVelocity())
                * Constants<FP>::lightVelocity() / (-Constants<FP>::electronCharge() * Constants<FP>::planck());

            coeffPhoton_probability = 1.0;
            coeffPair_probability = 1.0;

            int max_threads = OMP_GET_MAX_THREADS();

            rand_generator.resize(max_threads);
            distribution.resize(max_threads);
            for (int i = 0; i < max_threads; i++) {
                rand_generator[i].seed();
                distribution[i] = std::uniform_real_distribution<FP>(0.0, 1.0);
            }

            AvalanchePhotons.resize(max_threads);
            AvalancheParticles.resize(max_threads);
            afterAvalanchePhotons.resize(max_threads);
            afterAvalancheParticles.resize(max_threads);
            timeAvalanchePhotons.resize(max_threads);
            timeAvalancheParticles.resize(max_threads);
        }

        void disable_photon_emission()
        {
            this->coeffPhoton_probability = 0.0;
        }

        void enable_photon_emission()
        {
            this->coeffPhoton_probability = 1.0;
        }

        void disable_pair_production()
        {
            this->coeffPair_probability = 0.0;
        }

        void enable_pair_production()
        {
            this->coeffPair_probability = 1.0;
        }

        void processParticles(Ensemble3d* particles, TGrid* grid, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            for (int th = 0; th < max_threads; th++)
            {
                AvalanchePhotons[th].clear();
                AvalancheParticles[th].clear();
                afterAvalanchePhotons[th].clear();
                afterAvalancheParticles[th].clear();
            }

            if ((*particles)[Photon].size())
                HandlePhotons((*particles)[Photon], grid, timeStep);
            if ((*particles)[Electron].size())
                HandleParticles((*particles)[Electron], grid, timeStep);
            if ((*particles)[Positron].size())
                HandleParticles((*particles)[Positron], grid, timeStep);

            for (int th = 0; th < max_threads; th++)
            {
                for (int ind = 0; ind < afterAvalanchePhotons[th].size(); ind++)
                {
                    particles->addParticle(afterAvalanchePhotons[th][ind]);
                }
                for (int ind = 0; ind < afterAvalancheParticles[th].size(); ind++)
                {
                    particles->addParticle(afterAvalancheParticles[th][ind]);
                }
            }
        }

        // rewrite 2 borises
        void Boris(Particle3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void Boris(ParticleProxy3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void HandlePhotons(ParticleArray3d& particles, TGrid* grid, FP timeStep)
        {
#pragma omp parallel for
            for (int i = 0; i < particles.size(); i++)
            {
                int thread_id = OMP_GET_THREAD_NUM();

                FP3 pPos = particles[i].getPosition();
                FP3 e, b;

                e = grid->getE(pPos);
                b = grid->getB(pPos);

                AvalancheParticles[thread_id].clear();
                AvalanchePhotons[thread_id].clear();
                timeAvalancheParticles[thread_id].clear();
                timeAvalanchePhotons[thread_id].clear();
                AvalanchePhotons[thread_id].push_back(particles[i]);
                timeAvalanchePhotons[thread_id].push_back((FP)0.0);

                RunAvalanche(e, b, timeStep);

                for (int k = 0; k != AvalanchePhotons[thread_id].size(); k++)
                    if (AvalanchePhotons[thread_id][k].getGamma() != (FP)1.0)
                        afterAvalanchePhotons[thread_id].push_back(AvalanchePhotons[thread_id][k]);

                for (int k = 0; k != AvalancheParticles[thread_id].size(); k++)
                    afterAvalancheParticles[thread_id].push_back(AvalancheParticles[thread_id][k]);
            }

            particles.clear();
        }

        void HandleParticles(ParticleArray3d& particles, TGrid* grid, FP timeStep)
        {
#pragma omp parallel for
            for (int i = 0; i < particles.size(); i++)
            {
                int thread_id = OMP_GET_THREAD_NUM();

                FP3 pPos = particles[i].getPosition();
                FP3 e, b;

                e = grid->getE(pPos);
                b = grid->getB(pPos);

                AvalancheParticles[thread_id].clear();
                AvalanchePhotons[thread_id].clear();
                timeAvalancheParticles[thread_id].clear();
                timeAvalanchePhotons[thread_id].clear();
                AvalancheParticles[thread_id].push_back(particles[i]);
                timeAvalancheParticles[thread_id].push_back((FP)0.0);

                RunAvalanche(e, b, timeStep);

                for (int k = 0; k != AvalanchePhotons[thread_id].size(); k++)
                    if (AvalanchePhotons[thread_id][k].getGamma() != (FP)1.0)
                        afterAvalanchePhotons[thread_id].push_back(AvalanchePhotons[thread_id][k]);

                particles[i].setMomentum(AvalancheParticles[thread_id][0].getMomentum());
                particles[i].setPosition(AvalancheParticles[thread_id][0].getPosition());

                for (int k = 1; k != AvalancheParticles[thread_id].size(); k++)
                    afterAvalancheParticles[thread_id].push_back(AvalancheParticles[thread_id][k]);
            }
        }

        void RunAvalanche(const FP3& E, const FP3& B, double timeStep)
        {
            int thread_id = OMP_GET_THREAD_NUM();
            vector<Particle3d>& AvalancheParticles = this->AvalancheParticles[thread_id];
            vector<Particle3d>& AvalanchePhotons = this->AvalanchePhotons[thread_id];

            vector<FP>& timeAvalancheParticles = this->timeAvalancheParticles[thread_id];
            vector<FP>& timeAvalanchePhotons = this->timeAvalanchePhotons[thread_id];

            int countParticles = 0;
            int countPhotons = 0;

            while (countParticles != AvalancheParticles.size()
                || countPhotons != AvalanchePhotons.size())
            {
                for (int k = countParticles; k != AvalancheParticles.size(); k++)
                {
                    oneParticleStep(AvalancheParticles[k], E, B, timeAvalancheParticles[k], timeStep);
                    countParticles++;
                }
                for (int k = countPhotons; k != AvalanchePhotons.size(); k++)
                {
                    onePhotonStep(AvalanchePhotons[k], E, B, timeAvalanchePhotons[k], timeStep);
                    countPhotons++;
                }
            }
        }

        void oneParticleStep(Particle3d& particle, const FP3& E, const FP3& B, FP& time, double timeStep)
        {
            int thread_id = OMP_GET_THREAD_NUM();

            while (time < timeStep)
            {
                FP3 v = particle.getVelocity();
                FP H_eff = sqr(E + (1 / Constants<FP>::lightVelocity()) * VP(v, B))
                    - sqr(SP(E, v) / Constants<FP>::lightVelocity());
                if (H_eff < 0) H_eff = 0;
                H_eff = sqrt(H_eff);
                FP gamma = particle.getGamma();
                FP chi = gamma * H_eff / SchwingerField;
                FP rate = 0.0, dt = 2 * timeStep;
                if (chi > 0.0 && coeffPhoton_probability != 0.0)
                {
                    rate = compton.rate(chi);
                    dt = getDtParticle(particle, rate, chi);
                }

                if (dt + time > timeStep)
                {
                    Boris(particle, E, B, timeStep - time);
                    time = timeStep;
                }
                else
                {
                    Boris(particle, E, B, dt);
                    time += dt;
                    FP3 v = particle.getVelocity();
                    FP H_eff = sqr(E + (1 / Constants<FP>::lightVelocity()) * VP(v, B))
                        - sqr(SP(E, v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    FP gamma = particle.getGamma();
                    FP chi_new = gamma * H_eff / SchwingerField;

                    FP delta = Photon_Generator((chi + chi_new) / (FP)2.0);

                    Particle3d NewParticle;
                    NewParticle.setType(Photon);
                    NewParticle.setWeight(particle.getWeight());
                    NewParticle.setPosition(particle.getPosition());
                    NewParticle.setMomentum(delta * particle.getMomentum());

                    this->AvalanchePhotons[thread_id].push_back(NewParticle);
                    this->timeAvalanchePhotons[thread_id].push_back(time);
                    particle.setMomentum((1 - delta) * particle.getMomentum());
                }
            }
        }

        void onePhotonStep(Particle3d& particle, const FP3& E, const FP3& B, FP& time, double timeStep)
        {
            int thread_id = OMP_GET_THREAD_NUM();

            FP3 k_ = particle.getVelocity();
            k_ = (1 / k_.norm()) * k_; // normalized wave vector
            FP H_eff = sqrt(sqr(E + VP(k_, B)) - sqr(SP(E, k_)));
            FP gamma = particle.getMomentum().norm()
                / (Constants<FP>::electronMass() * Constants<FP>::lightVelocity());
            FP chi = gamma * H_eff / SchwingerField;

            FP rate = 0.0, dt = 2 * timeStep;
            if (chi > 0.0 && coeffPair_probability != 0.0)
            {
                rate = breit_wheeler.rate(chi);
                dt = getDtPhoton(particle, rate, chi, gamma);

            }

            if (dt + time > timeStep)
            {
                particle.setPosition(particle.getPosition()
                    + (timeStep - time) * Constants<FP>::lightVelocity() * k_);
                time = timeStep;
            }
            else
            {
                particle.setPosition(particle.getPosition()
                    + dt * Constants<FP>::lightVelocity() * k_);
                time += dt;
                FP delta = Pair_Generator(chi);

                Particle3d NewParticle;
                NewParticle.setType(Electron);
                NewParticle.setWeight(particle.getWeight());
                NewParticle.setPosition(particle.getPosition());
                NewParticle.setMomentum(delta * particle.getMomentum());

                this->AvalancheParticles[thread_id].push_back(NewParticle);
                this->timeAvalancheParticles[thread_id].push_back(time);

                NewParticle.setType(Positron);
                NewParticle.setMomentum((1 - delta) * particle.getMomentum());
                this->AvalancheParticles[thread_id].push_back(NewParticle);
                this->timeAvalancheParticles[thread_id].push_back(time);

                particle.setP((FP)0.0 * particle.getP());
                time = timeStep;
            }
        }

        FP getDtParticle(Particle3d& particle, FP rate, FP chi)
        {
            FP r = -log(random_number_omp());
            r *= (particle.getGamma()) / (chi);
            return r / rate;
        }

        FP Photon_Generator(FP chi)
        {
            FP r = random_number_omp();
            return compton.inv_cdf(r, chi);
        }


        FP getDtPhoton(Particle3d& particle, FP rate, FP chi, FP gamma)
        {
            FP r = -log(random_number_omp());
            r *= (gamma) / (chi);
            return r / rate;
        }

        FP Pair_Generator(FP chi)
        {
            FP r = random_number_omp();
            if (r < 0.5)
                return breit_wheeler.inv_cdf(r, chi);
            else
                return 1.0 - breit_wheeler.inv_cdf(1.0 - r, chi);
        }

        void operator()(ParticleProxy3d* particle, ValueField field, FP timeStep)
        {}

        void operator()(Particle3d* particle, ValueField field, FP timeStep)
        {
            ParticleProxy3d particleProxy(*particle);
            this->operator()(&particleProxy, field, timeStep);
        }


        Compton compton;
        Breit_wheeler breit_wheeler;

    private:

        FP random_number_omp()
        {
            int thread_id = OMP_GET_THREAD_NUM();
            return distribution[thread_id](rand_generator[thread_id]);
        }

        std::vector<std::default_random_engine> rand_generator;
        std::vector<std::uniform_real_distribution<FP>> distribution;

        std::vector<std::vector<FP>> timeAvalanchePhotons, timeAvalancheParticles;
        std::vector<std::vector<Particle3d>> AvalanchePhotons, AvalancheParticles;
        std::vector<std::vector<Particle3d>> afterAvalanchePhotons, afterAvalancheParticles;

        FP SchwingerField;

        FP coeffPhoton_probability, coeffPair_probability;
    };

    typedef Scalar_Fast_QED_v1<YeeGrid> Scalar_Fast_QED_Yee_v1;
    typedef Scalar_Fast_QED_v1<PSTDGrid> Scalar_Fast_QED_PSTD_v1;
    typedef Scalar_Fast_QED_v1<PSATDGrid> Scalar_Fast_QED_PSATD_v1;
    typedef Scalar_Fast_QED_v1<AnalyticalField> Scalar_Fast_QED_Analytical_v1;


    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////


    template <class TGrid>  // may be AnalyticalField or any Grid type
    class Scalar_Fast_QED_v2 : public ParticlePusher
    {
    public:

        Scalar_Fast_QED_v2() : compton(), breit_wheeler()
        {
            this->schwingerField = sqr(Constants<FP>::electronMass() * Constants<FP>::lightVelocity())
                * Constants<FP>::lightVelocity() / (-Constants<FP>::electronCharge() * Constants<FP>::planck());

            int max_threads = OMP_GET_MAX_THREADS();

            randGenerator.resize(max_threads);
            distribution.resize(max_threads);
            for (int i = 0; i < max_threads; i++) {
                randGenerator[i].seed();
                distribution[i] = std::uniform_real_distribution<FP>(0.0, 1.0);
            }

            this->generatedPhotons.resize(max_threads);
            this->generatedParticles.resize(max_threads);
        }

        void disablePhotonEmission() 
        {
            this->photonEmissionEnabled = false;
        }

        void enablePhotonEmission()
        {
            this->photonEmissionEnabled = true;
        }

        void disablePairProduction()
        {
            this->pairProductionEnabled = false;
        }

        void enablePairProduction()
        {
            this->pairProductionEnabled = true;
        }

        void processParticles(Ensemble3d* particles, TGrid* grid, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            if ((*particles)[Photon].size())
                HandlePhotons((*particles)[Photon], grid, timeStep);
            if ((*particles)[Electron].size())
                HandleParticles((*particles)[Electron], grid, timeStep);
            if ((*particles)[Positron].size())
                HandleParticles((*particles)[Positron], grid, timeStep);

            (*particles)[Photon].clear();

            for (int th = 0; th < max_threads; th++)
            {
                for (int ind = 0; ind < this->generatedPhotons[th].size(); ind++)
                {
                    particles->addParticle(this->generatedPhotons[th][ind]);
                }
                for (int ind = 0; ind < this->generatedParticles[th].size(); ind++)
                {
                    particles->addParticle(this->generatedParticles[th][ind]);
                }

                this->generatedPhotons[th].clear();
                this->generatedParticles[th].clear();
            }
        }

        // rewrite 2 borises
        void Boris(Particle3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void Boris(ParticleProxy3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void HandlePhotons(ParticleArray3d& photons, TGrid* grid, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            std::vector<std::vector<Particle3d>> avalancheParticlesThread(max_threads);
            std::vector<std::vector<FP>> timeAvalancheParticlesThread(max_threads);
            std::vector<std::vector<Particle3d>> avalanchePhotonsThread(max_threads);
            std::vector<std::vector<FP>> timeAvalanchePhotonsThread(max_threads);

#pragma omp parallel for
            for (int i = 0; i < photons.size(); i++)
            {
                FP3 pPos = photons[i].getPosition();
                FP3 e, b;

                e = grid->getE(pPos);
                b = grid->getB(pPos);

                int thread_id = OMP_GET_THREAD_NUM();

                std::vector<Particle3d>& avalancheParticles = avalancheParticlesThread[thread_id];
                std::vector<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[thread_id];
                std::vector<Particle3d>& avalanchePhotons = avalanchePhotonsThread[thread_id];
                std::vector<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[thread_id];

                // start avalanche with current photon
                avalanchePhotons.push_back(photons[i]);
                timeAvalanchePhotons.push_back((FP)0.0);

                RunAvalanche(e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push generated physical photons
                for (int k = 0; k != avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0)
                        this->generatedPhotons[thread_id].push_back(avalanchePhotons[k]);

                // push generated particles
                for (int k = 0; k != avalancheParticles.size(); k++)
                    this->generatedParticles[thread_id].push_back(avalancheParticles[k]);

                avalancheParticles.resize(0);
                timeAvalancheParticles.resize(0);
                avalanchePhotons.resize(0);
                timeAvalanchePhotons.resize(0);
            }
        }

        void HandleParticles(ParticleArray3d& particles, TGrid* grid, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            std::vector<std::vector<Particle3d>> avalancheParticlesThread(max_threads);
            std::vector<std::vector<FP>> timeAvalancheParticlesThread(max_threads);
            std::vector<std::vector<Particle3d>> avalanchePhotonsThread(max_threads);
            std::vector<std::vector<FP>> timeAvalanchePhotonsThread(max_threads);

#pragma omp parallel for
            for (int i = 0; i < particles.size(); i++)
            {
                FP3 pPos = particles[i].getPosition();
                FP3 e, b;

                e = grid->getE(pPos);
                b = grid->getB(pPos);

                int thread_id = OMP_GET_THREAD_NUM();

                std::vector<Particle3d>& avalancheParticles = avalancheParticlesThread[thread_id];
                std::vector<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[thread_id];
                std::vector<Particle3d>& avalanchePhotons = avalanchePhotonsThread[thread_id];
                std::vector<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[thread_id];

                // start avalanche with current particle
                avalancheParticles.push_back(particles[i]);
                timeAvalancheParticles.push_back((FP)0.0);

                RunAvalanche(e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push physical photons
                for (int k = 0; k != avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0) // include "removePhotonsBelow"
                        this->generatedPhotons[thread_id].push_back(avalanchePhotons[k]);

                // save new position and momentum of current particle
                particles[i].setMomentum(avalancheParticles[0].getMomentum());
                particles[i].setPosition(avalancheParticles[0].getPosition());
                // push generated particles
                for (int k = 1; k != avalancheParticles.size(); k++)
                    this->generatedParticles[thread_id].push_back(avalancheParticles[k]);

                avalancheParticles.resize(0);
                timeAvalancheParticles.resize(0);
                avalanchePhotons.resize(0);
                timeAvalanchePhotons.resize(0);
            }
        }
        
        void RunAvalanche(const FP3& E, const FP3& B, FP timeStep,
            std::vector<Particle3d>& avalancheParticles, std::vector<FP>& timeAvalancheParticles,
            std::vector<Particle3d>& avalanchePhotons, std::vector<FP>& timeAvalanchePhotons)
        {         
            int countProcessedParticles = 0;
            int countProcessedPhotons = 0;

            while (countProcessedParticles != avalancheParticles.size()
                || countProcessedPhotons != avalanchePhotons.size())
            {
                for (int k = countProcessedParticles; k != avalancheParticles.size(); k++)
                {
                    oneParticleStep(avalancheParticles[k], E, B, timeAvalancheParticles[k], timeStep,
                        avalanchePhotons, timeAvalanchePhotons);
                    countProcessedParticles++;
                }
                for (int k = countProcessedPhotons; k != avalanchePhotons.size(); k++)
                {
                    onePhotonStep(avalanchePhotons[k], E, B, timeAvalanchePhotons[k], timeStep,
                        avalancheParticles, timeAvalancheParticles);
                    countProcessedPhotons++;
                }
            }
        }

        void oneParticleStep(Particle3d& particle, const FP3& E, const FP3& B, FP& time, FP timeStep,
            std::vector<Particle3d>& avalanchePhotons, std::vector<FP>& timeAvalanchePhotons)
        {
            while (time < timeStep)
            {
                FP3 v = particle.getVelocity();

                FP H_eff = sqr(E + ((FP)1 / Constants<FP>::lightVelocity()) * VP(v, B))
                    - sqr(SP(E, v) / Constants<FP>::lightVelocity());
                if (H_eff < 0) H_eff = 0;
                H_eff = sqrt(H_eff);
                FP gamma = particle.getGamma();
                FP chi = gamma * H_eff / this->schwingerField;
                FP rate = (FP)0.0, dt = (FP)2*timeStep;
                if (chi > 0.0 && this->photonEmissionEnabled)
                {
                    rate = compton.rate(chi);
                    dt = getDt(rate, chi, gamma);
                }

                if (dt + time > timeStep)
                {
                    Boris(particle, E, B, timeStep - time);
                    time = timeStep;
                }
                else
                {
                    Boris(particle, E, B, dt);
                    time += dt;

                    FP3 v = particle.getVelocity();
                    FP H_eff = sqr(E + ((FP)1 / Constants<FP>::lightVelocity()) * VP(v, B))
                        - sqr(SP(E, v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    FP gamma = particle.getGamma();
                    FP chi_new = gamma * H_eff / this->schwingerField;

                    FP delta = photonGenerator((chi + chi_new) / (FP)2.0);
                    
                    Particle3d newPhoton;
                    newPhoton.setType(Photon);
                    newPhoton.setWeight(particle.getWeight());
                    newPhoton.setPosition(particle.getPosition());
                    newPhoton.setMomentum(delta * particle.getMomentum());

                    avalanchePhotons.push_back(newPhoton);
                    timeAvalanchePhotons.push_back(time);
                    particle.setMomentum(((FP)1 - delta) * particle.getMomentum());
                }
            }
        }

        void onePhotonStep(Particle3d& photon, const FP3& E, const FP3& B, FP& time, FP timeStep,
            std::vector<Particle3d>& avalancheParticles, std::vector<FP>& timeAvalancheParticles)
        {
            FP3 k_ = photon.getVelocity();
            k_ = ((FP)1 / k_.norm()) * k_; // normalized wave vector
            FP H_eff = sqrt(sqr(E + VP(k_, B)) - sqr(SP(E, k_)));
            FP gamma = photon.getMomentum().norm()
                / (Constants<FP>::electronMass() * Constants<FP>::lightVelocity());
            FP chi = gamma * H_eff / this->schwingerField;

            FP rate = 0.0, dt = (FP)2 * timeStep;
            if (chi > 0.0 && this->pairProductionEnabled)
            {
                rate = breit_wheeler.rate(chi);
                dt = getDt(rate, chi, gamma);
            }

            if (dt + time > timeStep)
            {
                photon.setPosition(photon.getPosition()
                    + (timeStep - time) * Constants<FP>::lightVelocity() * k_);
                time = timeStep;
            }
            else
            {
                photon.setPosition(photon.getPosition()
                    + dt * Constants<FP>::lightVelocity() * k_);
                time += dt;
                FP delta = pairGenerator(chi);

                Particle3d newParticle;
                newParticle.setType(Electron);
                newParticle.setWeight(photon.getWeight());
                newParticle.setPosition(photon.getPosition());
                newParticle.setMomentum(delta * photon.getMomentum());

                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);

                newParticle.setType(Positron);
                newParticle.setMomentum(((FP)1 - delta) * photon.getMomentum());
                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);

                photon.setP((FP)0.0 * photon.getP());  // mark old photon as deleted
                time = timeStep;
            }
        }
        
        FP getDt(FP rate, FP chi, FP gamma)
        {
            FP r = -log(random_number_omp());
            r *= gamma / chi;
            return r / rate;
        }
        
        FP photonGenerator(FP chi)
        {
            FP r = random_number_omp();
            return compton.inv_cdf(r, chi);
        }
        
        FP pairGenerator(FP chi)
        {
            FP r = random_number_omp();
            if (r < (FP)0.5)
                return breit_wheeler.inv_cdf(r, chi);
            else
                return (FP)1.0 - breit_wheeler.inv_cdf((FP)1.0 - r, chi);
        }

        void operator()(ParticleProxy3d* particle, ValueField field, FP timeStep)
        {}

        void operator()(Particle3d* particle, ValueField field, FP timeStep)
        {
            ParticleProxy3d particleProxy(*particle);
            this->operator()(&particleProxy, field, timeStep);
        }

        Compton compton;
        Breit_wheeler breit_wheeler;

    private:

        FP random_number_omp()
        {
            int thread_id = OMP_GET_THREAD_NUM();
            return distribution[thread_id](randGenerator[thread_id]);
        }

        std::vector<std::default_random_engine> randGenerator;
        std::vector<std::uniform_real_distribution<FP>> distribution;

        std::vector<std::vector<Particle3d>> generatedPhotons, generatedParticles;

        FP schwingerField;
        
        bool photonEmissionEnabled = true, pairProductionEnabled = true;
    };

    typedef Scalar_Fast_QED_v2<YeeGrid> Scalar_Fast_QED_Yee_v2;
    typedef Scalar_Fast_QED_v2<PSTDGrid> Scalar_Fast_QED_PSTD_v2;
    typedef Scalar_Fast_QED_v2<PSATDGrid> Scalar_Fast_QED_PSATD_v2;
    typedef Scalar_Fast_QED_v2<AnalyticalField> Scalar_Fast_QED_Analytical_v2;


    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////


    template <class TGrid>  // may be AnalyticalField or any Grid type
    class Scalar_Fast_QED_v3 : public ParticlePusher
    {
    public:

        Scalar_Fast_QED_v3() : compton(), breit_wheeler()
        {
            this->schwingerField = sqr(Constants<FP>::electronMass() * Constants<FP>::lightVelocity())
                * Constants<FP>::lightVelocity() / (-Constants<FP>::electronCharge() * Constants<FP>::planck());

            int max_threads = OMP_GET_MAX_THREADS();

            randGenerator.resize(max_threads);
            distribution.resize(max_threads);
            for (int i = 0; i < max_threads; i++) {
                randGenerator[i].seed();
                distribution[i] = std::uniform_real_distribution<FP>(0.0, 1.0);
            }

            this->generatedPhotons.resize(max_threads);
            this->generatedParticles.resize(max_threads);
        }

        void disablePhotonEmission()
        {
            this->photonEmissionEnabled = false;
        }

        void enablePhotonEmission()
        {
            this->photonEmissionEnabled = true;
        }

        void disablePairProduction()
        {
            this->pairProductionEnabled = false;
        }

        void enablePairProduction()
        {
            this->pairProductionEnabled = true;
        }

        void processParticles(Ensemble3d* particles,
            const std::vector<std::vector<FP3>>& e, const std::vector<std::vector<FP3>>& b, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            if ((*particles)[Photon].size())
                HandlePhotons((*particles)[Photon], e[0], b[0], timeStep);
            if ((*particles)[Electron].size())
                HandleParticles((*particles)[Electron], e[1], b[1], timeStep);
            if ((*particles)[Positron].size())
                HandleParticles((*particles)[Positron], e[2], b[2], timeStep);

            (*particles)[Photon].clear();

            for (int th = 0; th < max_threads; th++)
            {
                for (int ind = 0; ind < this->generatedPhotons[th].size(); ind++)
                {
                    particles->addParticle(this->generatedPhotons[th][ind]);
                }
                for (int ind = 0; ind < this->generatedParticles[th].size(); ind++)
                {
                    particles->addParticle(this->generatedParticles[th][ind]);
                }

                this->generatedPhotons[th].clear();
                this->generatedParticles[th].clear();
            }
        }

        // rewrite 2 borises
        void Boris(Particle3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void Boris(ParticleProxy3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void HandlePhotons(ParticleArray3d& photons, const std::vector<FP3>& ve,
            const std::vector<FP3>& vb, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            std::vector<std::vector<Particle3d>> avalancheParticlesThread(max_threads);
            std::vector<std::vector<FP>> timeAvalancheParticlesThread(max_threads);
            std::vector<std::vector<Particle3d>> avalanchePhotonsThread(max_threads);
            std::vector<std::vector<FP>> timeAvalanchePhotonsThread(max_threads);

#pragma omp parallel for
            for (int i = 0; i < photons.size(); i++)
            {
                FP3 pPos = photons[i].getPosition();
                FP3 e = ve[i], b = vb[i];

                int thread_id = OMP_GET_THREAD_NUM();

                std::vector<Particle3d>& avalancheParticles = avalancheParticlesThread[thread_id];
                std::vector<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[thread_id];
                std::vector<Particle3d>& avalanchePhotons = avalanchePhotonsThread[thread_id];
                std::vector<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[thread_id];

                // start avalanche with current photon
                avalanchePhotons.push_back(photons[i]);
                timeAvalanchePhotons.push_back((FP)0.0);

                RunAvalanche(e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push generated physical photons
                for (int k = 0; k != avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0)
                        this->generatedPhotons[thread_id].push_back(avalanchePhotons[k]);

                // push generated particles
                for (int k = 0; k != avalancheParticles.size(); k++)
                    this->generatedParticles[thread_id].push_back(avalancheParticles[k]);

                avalancheParticles.resize(0);
                timeAvalancheParticles.resize(0);
                avalanchePhotons.resize(0);
                timeAvalanchePhotons.resize(0);
            }
        }

        void HandleParticles(ParticleArray3d& particles, const std::vector<FP3>& ve,
            const std::vector<FP3>& vb, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            std::vector<std::vector<Particle3d>> avalancheParticlesThread(max_threads);
            std::vector<std::vector<FP>> timeAvalancheParticlesThread(max_threads);
            std::vector<std::vector<Particle3d>> avalanchePhotonsThread(max_threads);
            std::vector<std::vector<FP>> timeAvalanchePhotonsThread(max_threads);

#pragma omp parallel for
            for (int i = 0; i < particles.size(); i++)
            {
                FP3 pPos = particles[i].getPosition();
                FP3 e = ve[i], b = vb[i];

                int thread_id = OMP_GET_THREAD_NUM();

                std::vector<Particle3d>& avalancheParticles = avalancheParticlesThread[thread_id];
                std::vector<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[thread_id];
                std::vector<Particle3d>& avalanchePhotons = avalanchePhotonsThread[thread_id];
                std::vector<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[thread_id];

                // start avalanche with current particle
                avalancheParticles.push_back(particles[i]);
                timeAvalancheParticles.push_back((FP)0.0);

                RunAvalanche(e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push physical photons
                for (int k = 0; k != avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0) // include "removePhotonsBelow"
                        this->generatedPhotons[thread_id].push_back(avalanchePhotons[k]);

                // save new position and momentum of current particle
                particles[i].setMomentum(avalancheParticles[0].getMomentum());
                particles[i].setPosition(avalancheParticles[0].getPosition());
                // push generated particles
                for (int k = 1; k != avalancheParticles.size(); k++)
                    this->generatedParticles[thread_id].push_back(avalancheParticles[k]);

                avalancheParticles.resize(0);
                timeAvalancheParticles.resize(0);
                avalanchePhotons.resize(0);
                timeAvalanchePhotons.resize(0);
            }
        }

        void RunAvalanche(const FP3& e, const FP3& b, FP timeStep,
            std::vector<Particle3d>& avalancheParticles, std::vector<FP>& timeAvalancheParticles,
            std::vector<Particle3d>& avalanchePhotons, std::vector<FP>& timeAvalanchePhotons)
        {
            int countProcessedParticles = 0;
            int countProcessedPhotons = 0;
        
            while (countProcessedParticles != avalancheParticles.size()
                || countProcessedPhotons != avalanchePhotons.size())
            {
                for (int k = countProcessedParticles; k != avalancheParticles.size(); k++)
                {
                    oneParticleStep(avalancheParticles[k], e, b, timeAvalancheParticles[k], timeStep,
                        avalanchePhotons, timeAvalanchePhotons);
                    countProcessedParticles++;
                }
                for (int k = countProcessedPhotons; k != avalanchePhotons.size(); k++)
                {
                    onePhotonStep(avalanchePhotons[k], e, b, timeAvalanchePhotons[k], timeStep,
                        avalancheParticles, timeAvalancheParticles);
                    countProcessedPhotons++;
                }
            }
        }

        void oneParticleStep(Particle3d& particle, const FP3& e, const FP3& b, FP& time, FP timeStep,
            std::vector<Particle3d>& avalanchePhotons, std::vector<FP>& timeAvalanchePhotons)
        {
            FP invLightVelocity = (FP)1 / Constants<FP>::lightVelocity();
            FP invSchwingerField = (FP)1 / this->schwingerField;
        
            while (time < timeStep)
            {
                FP3 v = particle.getVelocity();
        
                FP H_eff = sqr(e + invLightVelocity * VP(v, b))
                    - sqr(SP(e, v) / Constants<FP>::lightVelocity());
                if (H_eff < 0) H_eff = 0;
                H_eff = sqrt(H_eff);
                FP gamma = particle.getGamma();
                FP chi = gamma * H_eff * invSchwingerField;
                FP rate = (FP)0.0, dt = (FP)2 * timeStep;
                if (chi > 0.0 && this->photonEmissionEnabled)
                {
                    rate = compton.rate(chi);
                    dt = getDt(rate, chi, gamma);
                }
        
                if (dt + time > timeStep)
                {
                    Boris(particle, e, b, timeStep - time);
                    time = timeStep;
                }
                else
                {
                    Boris(particle, e, b, dt);
                    time += dt;
        
                    FP3 v = particle.getVelocity();
                    FP H_eff = sqr(e + invLightVelocity * VP(v, b))
                        - sqr(SP(e, v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    FP gamma = particle.getGamma();
                    FP chi_new = gamma * H_eff * invSchwingerField;
        
                    FP delta = photonGenerator((chi + chi_new) * (FP)0.5);
        
                    Particle3d newPhoton;
                    newPhoton.setType(Photon);
                    newPhoton.setWeight(particle.getWeight());
                    newPhoton.setPosition(particle.getPosition());
                    newPhoton.setMomentum(delta * particle.getMomentum());
        
                    avalanchePhotons.push_back(newPhoton);
                    timeAvalanchePhotons.push_back(time);
                    particle.setMomentum(((FP)1 - delta) * particle.getMomentum());
                }
            }
        }

        void onePhotonStep(Particle3d& photon, const FP3& e, const FP3& b, FP& time, FP timeStep,
            std::vector<Particle3d>& avalancheParticles, std::vector<FP>& timeAvalancheParticles)
        {
            FP3 k_ = photon.getVelocity();
            k_ = ((FP)1 / k_.norm()) * k_; // normalized wave vector
            FP H_eff = sqrt(sqr(e + VP(k_, b)) - sqr(SP(e, k_)));
            FP gamma = photon.getMomentum().norm()
                / (Constants<FP>::electronMass() * Constants<FP>::lightVelocity());
            FP chi = gamma * H_eff / this->schwingerField;
        
            FP rate = 0.0, dt = (FP)2 * timeStep;
            if (chi > 0.0 && this->pairProductionEnabled)
            {
                rate = breit_wheeler.rate(chi);
                dt = getDt(rate, chi, gamma);
            }
        
            if (dt + time > timeStep)
            {
                photon.setPosition(photon.getPosition()
                    + (timeStep - time) * Constants<FP>::lightVelocity() * k_);
                time = timeStep;
            }
            else
            {
                photon.setPosition(photon.getPosition()
                    + dt * Constants<FP>::lightVelocity() * k_);
                time += dt;
                FP delta = pairGenerator(chi);
        
                Particle3d newParticle;
                newParticle.setType(Electron);
                newParticle.setWeight(photon.getWeight());
                newParticle.setPosition(photon.getPosition());
                newParticle.setMomentum(delta * photon.getMomentum());
        
                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);
        
                newParticle.setType(Positron);
                newParticle.setMomentum(((FP)1 - delta) * photon.getMomentum());
                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);
        
                photon.setP((FP)0.0 * photon.getP());  // mark old photon as deleted
                time = timeStep;
            }
        }

        FP getDt(FP rate, FP chi, FP gamma)
        {
            FP r = -log(random_number_omp());
            r *= gamma / chi;
            return r / rate;
        }

        FP photonGenerator(FP chi)
        {
            FP r = random_number_omp();
            return compton.inv_cdf(r, chi);
        }

        FP pairGenerator(FP chi)
        {
            FP r = random_number_omp();
            if (r < (FP)0.5)
                return breit_wheeler.inv_cdf(r, chi);
            else
                return (FP)1.0 - breit_wheeler.inv_cdf((FP)1.0 - r, chi);
        }

        void operator()(ParticleProxy3d* particle, ValueField field, FP timeStep)
        {}

        void operator()(Particle3d* particle, ValueField field, FP timeStep)
        {
            ParticleProxy3d particleProxy(*particle);
            this->operator()(&particleProxy, field, timeStep);
        }

        Compton compton;
        Breit_wheeler breit_wheeler;

    private:

        FP random_number_omp()
        {
            int thread_id = OMP_GET_THREAD_NUM();
            return distribution[thread_id](randGenerator[thread_id]);
        }

        std::vector<std::default_random_engine> randGenerator;
        std::vector<std::uniform_real_distribution<FP>> distribution;

        std::vector<std::vector<Particle3d>> generatedPhotons, generatedParticles;

        FP schwingerField;

        bool photonEmissionEnabled = true, pairProductionEnabled = true;
    };

    typedef Scalar_Fast_QED_v3<YeeGrid> Scalar_Fast_QED_Yee_v3;
    typedef Scalar_Fast_QED_v3<PSTDGrid> Scalar_Fast_QED_PSTD_v3;
    typedef Scalar_Fast_QED_v3<PSATDGrid> Scalar_Fast_QED_PSATD_v3;
    typedef Scalar_Fast_QED_v3<AnalyticalField> Scalar_Fast_QED_Analytical_v3;


    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////


    template <class TGrid>  // may be AnalyticalField or any Grid type
    class Scalar_Fast_QED_v4 : public ParticlePusher
    {
    public:

        Scalar_Fast_QED_v4() : compton(), breit_wheeler()
        {
            this->schwingerField = sqr(Constants<FP>::electronMass() * Constants<FP>::lightVelocity())
                * Constants<FP>::lightVelocity() / (-Constants<FP>::electronCharge() * Constants<FP>::planck());

            int max_threads = OMP_GET_MAX_THREADS();

            randGenerator.resize(max_threads);
            distribution.resize(max_threads);
            for (int i = 0; i < max_threads; i++) {
                randGenerator[i].seed();
                distribution[i] = std::uniform_real_distribution<FP>(0.0, 1.0);
            }

            this->generatedPhotons.resize(max_threads);
            this->generatedParticles.resize(max_threads);
        }

        void disablePhotonEmission()
        {
            this->photonEmissionEnabled = false;
        }

        void enablePhotonEmission()
        {
            this->photonEmissionEnabled = true;
        }

        void disablePairProduction()
        {
            this->pairProductionEnabled = false;
        }

        void enablePairProduction()
        {
            this->pairProductionEnabled = true;
        }

        void processParticles(Ensemble3d* particles,
            const std::vector<std::vector<FP3>>& e, const std::vector<std::vector<FP3>>& b, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            if ((*particles)[Photon].size())
                HandlePhotons((*particles)[Photon], e[0], b[0], timeStep);
            if ((*particles)[Electron].size())
                HandleParticles((*particles)[Electron], e[1], b[1], timeStep);
            if ((*particles)[Positron].size())
                HandleParticles((*particles)[Positron], e[2], b[2], timeStep);

            (*particles)[Photon].clear();

            for (int th = 0; th < max_threads; th++)
            {
                for (int ind = 0; ind < this->generatedPhotons[th].size(); ind++)
                {
                    particles->addParticle(this->generatedPhotons[th][ind]);
                }
                for (int ind = 0; ind < this->generatedParticles[th].size(); ind++)
                {
                    particles->addParticle(this->generatedParticles[th][ind]);
                }

                this->generatedPhotons[th].clear();
                this->generatedParticles[th].clear();
            }
        }

        // rewrite 2 borises
        void Boris(Particle3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void Boris(ParticleProxy3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void HandlePhotons(ParticleArray3d& photons, const std::vector<FP3>& ve,
            const std::vector<FP3>& vb, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            std::vector<std::vector<Particle3d>> avalancheParticlesThread(max_threads);
            std::vector<std::vector<FP>> timeAvalancheParticlesThread(max_threads);
            std::vector<std::vector<Particle3d>> avalanchePhotonsThread(max_threads);
            std::vector<std::vector<FP>> timeAvalanchePhotonsThread(max_threads);

            const int reserveValue = 10;
            for (int thr = 0; thr < max_threads; thr++) {
                avalancheParticlesThread[thr].reserve(reserveValue);
                timeAvalancheParticlesThread[thr].reserve(reserveValue);
                avalanchePhotonsThread[thr].reserve(reserveValue);
                timeAvalanchePhotonsThread[thr].reserve(reserveValue);
            }

#pragma omp parallel for
            for (int i = 0; i < photons.size(); i++)
            {
                FP3 pPos = photons[i].getPosition();
                FP3 e = ve[i], b = vb[i];

                int thread_id = OMP_GET_THREAD_NUM();

                std::vector<Particle3d>& avalancheParticles = avalancheParticlesThread[thread_id];
                std::vector<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[thread_id];
                std::vector<Particle3d>& avalanchePhotons = avalanchePhotonsThread[thread_id];
                std::vector<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[thread_id];

                // start avalanche with current photon
                avalanchePhotons.push_back(photons[i]);
                timeAvalanchePhotons.push_back((FP)0.0);

                RunAvalanche(e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push generated physical photons
                for (int k = 0; k != avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0)
                        this->generatedPhotons[thread_id].push_back(avalanchePhotons[k]);

                // push generated particles
                for (int k = 0; k != avalancheParticles.size(); k++)
                    this->generatedParticles[thread_id].push_back(avalancheParticles[k]);

                avalancheParticles.clear();
                timeAvalancheParticles.clear();
                avalanchePhotons.clear();
                timeAvalanchePhotons.clear();
            }
        }

        void HandleParticles(ParticleArray3d& particles, const std::vector<FP3>& ve,
            const std::vector<FP3>& vb, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            std::vector<std::vector<Particle3d>> avalancheParticlesThread(max_threads);
            std::vector<std::vector<FP>> timeAvalancheParticlesThread(max_threads);
            std::vector<std::vector<Particle3d>> avalanchePhotonsThread(max_threads);
            std::vector<std::vector<FP>> timeAvalanchePhotonsThread(max_threads);

            const int reserveValue = 10;
            for (int thr = 0; thr < max_threads; thr++) {
                avalancheParticlesThread[thr].reserve(reserveValue);
                timeAvalancheParticlesThread[thr].reserve(reserveValue);
                avalanchePhotonsThread[thr].reserve(reserveValue);
                timeAvalanchePhotonsThread[thr].reserve(reserveValue);
            }

#pragma omp parallel for
            for (int i = 0; i < particles.size(); i++)
            {
                FP3 pPos = particles[i].getPosition();
                FP3 e = ve[i], b = vb[i];

                int thread_id = OMP_GET_THREAD_NUM();

                std::vector<Particle3d>& avalancheParticles = avalancheParticlesThread[thread_id];
                std::vector<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[thread_id];
                std::vector<Particle3d>& avalanchePhotons = avalanchePhotonsThread[thread_id];
                std::vector<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[thread_id];

                // start avalanche with current particle
                avalancheParticles.push_back(particles[i]);
                timeAvalancheParticles.push_back((FP)0.0);

                RunAvalanche(e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push physical photons
                for (int k = 0; k != avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0) // include "removePhotonsBelow"
                        this->generatedPhotons[thread_id].push_back(avalanchePhotons[k]);

                // save new position and momentum of current particle
                particles[i].setMomentum(avalancheParticles[0].getMomentum());
                particles[i].setPosition(avalancheParticles[0].getPosition());
                // push generated particles
                for (int k = 1; k != avalancheParticles.size(); k++)
                    this->generatedParticles[thread_id].push_back(avalancheParticles[k]);

                avalancheParticles.clear();
                timeAvalancheParticles.clear();
                avalanchePhotons.clear();
                timeAvalanchePhotons.clear();
            }
        }

        void RunAvalanche(const FP3& e, const FP3& b, FP timeStep,
            std::vector<Particle3d>& avalancheParticles, std::vector<FP>& timeAvalancheParticles,
            std::vector<Particle3d>& avalanchePhotons, std::vector<FP>& timeAvalanchePhotons)
        {
            int countProcessedParticles = 0;
            int countProcessedPhotons = 0;

            int avalancheParticlesSize = avalancheParticles.size();
            int avalanchePhotonsSize = avalanchePhotons.size();

            while (countProcessedParticles != avalancheParticlesSize
                || countProcessedPhotons != avalanchePhotonsSize)
            {
                for (int k = countProcessedParticles; k != avalancheParticlesSize; k++)
                {
                    oneParticleStep(avalancheParticles[k], e, b, timeAvalancheParticles[k], timeStep,
                        avalanchePhotons, timeAvalanchePhotons);
                }
                avalanchePhotonsSize = avalanchePhotons.size();
                countProcessedParticles = avalancheParticles.size();

                for (int k = countProcessedPhotons; k != avalanchePhotonsSize; k++)
                {
                    onePhotonStep(avalanchePhotons[k], e, b, timeAvalanchePhotons[k], timeStep,
                        avalancheParticles, timeAvalancheParticles);
                }
                avalancheParticlesSize = avalancheParticles.size();
                countProcessedPhotons = avalanchePhotons.size();
            }
        }

        void oneParticleStep(Particle3d& particle, const FP3& e, const FP3& b, FP& time, FP timeStep,
            std::vector<Particle3d>& avalanchePhotons, std::vector<FP>& timeAvalanchePhotons)
        {
            FP invLightVelocity = (FP)1 / Constants<FP>::lightVelocity();
            FP invSchwingerField = (FP)1 / this->schwingerField;

            while (time < timeStep)
            {
                FP3 v = particle.getVelocity();

                FP H_eff = sqr(e + invLightVelocity * VP(v, b))
                    - sqr(SP(e, v) / Constants<FP>::lightVelocity());
                if (H_eff < 0) H_eff = 0;
                H_eff = sqrt(H_eff);
                FP gamma = particle.getGamma();
                FP chi = gamma * H_eff * invSchwingerField;
                FP rate = (FP)0.0, dt = (FP)2 * timeStep;
                if (chi > 0.0 && this->photonEmissionEnabled)
                {
                    rate = compton.rate(chi);
                    dt = getDt(rate, chi, gamma);
                }

                if (dt + time > timeStep)
                {
                    Boris(particle, e, b, timeStep - time);
                    time = timeStep;
                }
                else
                {
                    Boris(particle, e, b, dt);
                    time += dt;

                    FP3 v = particle.getVelocity();
                    FP H_eff = sqr(e + invLightVelocity * VP(v, b))
                        - sqr(SP(e, v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    FP gamma = particle.getGamma();
                    FP chi_new = gamma * H_eff * invSchwingerField;

                    FP delta = photonGenerator((chi + chi_new) * (FP)0.5);

                    Particle3d newPhoton;
                    newPhoton.setType(Photon);
                    newPhoton.setWeight(particle.getWeight());
                    newPhoton.setPosition(particle.getPosition());
                    newPhoton.setMomentum(delta * particle.getMomentum());

                    avalanchePhotons.push_back(newPhoton);
                    timeAvalanchePhotons.push_back(time);
                    particle.setMomentum(((FP)1 - delta) * particle.getMomentum());
                }
            }
        }

        void onePhotonStep(Particle3d& photon, const FP3& e, const FP3& b, FP& time, FP timeStep,
            std::vector<Particle3d>& avalancheParticles, std::vector<FP>& timeAvalancheParticles)
        {
            FP3 k_ = photon.getVelocity();
            k_ = ((FP)1 / k_.norm()) * k_; // normalized wave vector
            FP H_eff = sqrt(sqr(e + VP(k_, b)) - sqr(SP(e, k_)));
            FP gamma = photon.getMomentum().norm()
                / (Constants<FP>::electronMass() * Constants<FP>::lightVelocity());
            FP chi = gamma * H_eff / this->schwingerField;

            FP rate = 0.0, dt = (FP)2 * timeStep;
            if (chi > 0.0 && this->pairProductionEnabled)
            {
                rate = breit_wheeler.rate(chi);
                dt = getDt(rate, chi, gamma);
            }

            if (dt + time > timeStep)
            {
                photon.setPosition(photon.getPosition()
                    + (timeStep - time) * Constants<FP>::lightVelocity() * k_);
                time = timeStep;
            }
            else
            {
                photon.setPosition(photon.getPosition()
                    + dt * Constants<FP>::lightVelocity() * k_);
                time += dt;
                FP delta = pairGenerator(chi);

                Particle3d newParticle;
                newParticle.setType(Electron);
                newParticle.setWeight(photon.getWeight());
                newParticle.setPosition(photon.getPosition());
                newParticle.setMomentum(delta * photon.getMomentum());

                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);

                newParticle.setType(Positron);
                newParticle.setMomentum(((FP)1 - delta) * photon.getMomentum());
                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);

                photon.setP((FP)0.0 * photon.getP());  // mark old photon as deleted
                time = timeStep;
            }
        }

        FP getDt(FP rate, FP chi, FP gamma)
        {
            FP r = -log(random_number_omp());
            r *= gamma / chi;
            return r / rate;
        }

        FP photonGenerator(FP chi)
        {
            FP r = random_number_omp();
            return compton.inv_cdf(r, chi);
        }

        FP pairGenerator(FP chi)
        {
            FP r = random_number_omp();
            if (r < (FP)0.5)
                return breit_wheeler.inv_cdf(r, chi);
            else
                return (FP)1.0 - breit_wheeler.inv_cdf((FP)1.0 - r, chi);
        }

        void operator()(ParticleProxy3d* particle, ValueField field, FP timeStep)
        {}

        void operator()(Particle3d* particle, ValueField field, FP timeStep)
        {
            ParticleProxy3d particleProxy(*particle);
            this->operator()(&particleProxy, field, timeStep);
        }

        Compton compton;
        Breit_wheeler breit_wheeler;

    private:

        FP random_number_omp()
        {
            int thread_id = OMP_GET_THREAD_NUM();
            return distribution[thread_id](randGenerator[thread_id]);
        }

        std::vector<std::default_random_engine> randGenerator;
        std::vector<std::uniform_real_distribution<FP>> distribution;

        std::vector<std::vector<Particle3d>> generatedPhotons, generatedParticles;

        FP schwingerField;

        bool photonEmissionEnabled = true, pairProductionEnabled = true;
    };

    typedef Scalar_Fast_QED_v4<YeeGrid> Scalar_Fast_QED_Yee_v4;
    typedef Scalar_Fast_QED_v4<PSTDGrid> Scalar_Fast_QED_PSTD_v4;
    typedef Scalar_Fast_QED_v4<PSATDGrid> Scalar_Fast_QED_PSATD_v4;
    typedef Scalar_Fast_QED_v4<AnalyticalField> Scalar_Fast_QED_Analytical_v4;


    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////


    template <class TGrid>  // may be AnalyticalField or any Grid type
    class Scalar_Fast_QED_v5 : public ParticlePusher
    {
    public:

        template <class T>
        struct myVector {
            int n = 0, capacity = 10;
            std::vector<T> data;

            myVector(): data(capacity) {}

            T& operator[](int i) { return data[i]; }
            const T& operator[](int i) const { return data[i]; }

            const int& size() const { return n; }
            void clear() { n = 0; }

            void push_back(const T& value) {
                if (n == capacity) {
                    capacity *= 2;
                    data.resize(capacity);
                }
                data[n++] = value;
            }

        };

        Scalar_Fast_QED_v5() : compton(), breit_wheeler()
        {
            this->schwingerField = sqr(Constants<FP>::electronMass() * Constants<FP>::lightVelocity())
                * Constants<FP>::lightVelocity() / (-Constants<FP>::electronCharge() * Constants<FP>::planck());

            int max_threads = OMP_GET_MAX_THREADS();

            randGenerator.resize(max_threads);
            distribution.resize(max_threads);
            for (int i = 0; i < max_threads; i++) {
                randGenerator[i].seed();
                distribution[i] = std::uniform_real_distribution<FP>(0.0, 1.0);
            }

            this->generatedPhotons.resize(max_threads);
            this->generatedParticles.resize(max_threads);
        }

        void disablePhotonEmission()
        {
            this->photonEmissionEnabled = false;
        }

        void enablePhotonEmission()
        {
            this->photonEmissionEnabled = true;
        }

        void disablePairProduction()
        {
            this->pairProductionEnabled = false;
        }

        void enablePairProduction()
        {
            this->pairProductionEnabled = true;
        }

        void processParticles(Ensemble3d* particles,
            const std::vector<std::vector<FP3>>& e, const std::vector<std::vector<FP3>>& b, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            if ((*particles)[Photon].size())
                HandlePhotons((*particles)[Photon], e[0], b[0], timeStep);
            if ((*particles)[Electron].size())
                HandleParticles((*particles)[Electron], e[1], b[1], timeStep);
            if ((*particles)[Positron].size())
                HandleParticles((*particles)[Positron], e[2], b[2], timeStep);

            (*particles)[Photon].clear();

            for (int th = 0; th < max_threads; th++)
            {
                for (int ind = 0; ind < this->generatedPhotons[th].size(); ind++)
                {
                    particles->addParticle(this->generatedPhotons[th][ind]);
                }
                for (int ind = 0; ind < this->generatedParticles[th].size(); ind++)
                {
                    particles->addParticle(this->generatedParticles[th][ind]);
                }

                this->generatedPhotons[th].clear();
                this->generatedParticles[th].clear();
            }
        }

        // rewrite 2 borises
        void Boris(Particle3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void Boris(ParticleProxy3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void HandlePhotons(ParticleArray3d& photons, const std::vector<FP3>& ve,
            const std::vector<FP3>& vb, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            std::vector<myVector<Particle3d>> avalancheParticlesThread(max_threads);
            std::vector<myVector<FP>> timeAvalancheParticlesThread(max_threads);
            std::vector<myVector<Particle3d>> avalanchePhotonsThread(max_threads);
            std::vector<myVector<FP>> timeAvalanchePhotonsThread(max_threads);

#pragma omp parallel for
            for (int i = 0; i < photons.size(); i++)
            {
                FP3 pPos = photons[i].getPosition();
                FP3 e = ve[i], b = vb[i];

                int thread_id = OMP_GET_THREAD_NUM();

                myVector<Particle3d>& avalancheParticles = avalancheParticlesThread[thread_id];
                myVector<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[thread_id];
                myVector<Particle3d>& avalanchePhotons = avalanchePhotonsThread[thread_id];
                myVector<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[thread_id];

                // start avalanche with current photon
                avalanchePhotons.push_back(photons[i]);
                timeAvalanchePhotons.push_back((FP)0.0);

                RunAvalanche(e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push generated physical photons
                for (int k = 0; k != avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0)
                        this->generatedPhotons[thread_id].push_back(avalanchePhotons[k]);

                // push generated particles
                for (int k = 0; k != avalancheParticles.size(); k++)
                    this->generatedParticles[thread_id].push_back(avalancheParticles[k]);

                avalancheParticles.clear();
                timeAvalancheParticles.clear();
                avalanchePhotons.clear();
                timeAvalanchePhotons.clear();
            }
        }

        void HandleParticles(ParticleArray3d& particles, const std::vector<FP3>& ve,
            const std::vector<FP3>& vb, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            std::vector<myVector<Particle3d>> avalancheParticlesThread(max_threads);
            std::vector<myVector<FP>> timeAvalancheParticlesThread(max_threads);
            std::vector<myVector<Particle3d>> avalanchePhotonsThread(max_threads);
            std::vector<myVector<FP>> timeAvalanchePhotonsThread(max_threads);

#pragma omp parallel for
            for (int i = 0; i < particles.size(); i++)
            {
                FP3 pPos = particles[i].getPosition();
                FP3 e = ve[i], b = vb[i];

                int thread_id = OMP_GET_THREAD_NUM();

                myVector<Particle3d>& avalancheParticles = avalancheParticlesThread[thread_id];
                myVector<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[thread_id];
                myVector<Particle3d>& avalanchePhotons = avalanchePhotonsThread[thread_id];
                myVector<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[thread_id];

                // start avalanche with current particle
                avalancheParticles.push_back(particles[i]);
                timeAvalancheParticles.push_back((FP)0.0);

                RunAvalanche(e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push physical photons
                for (int k = 0; k != avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0) // include "removePhotonsBelow"
                        this->generatedPhotons[thread_id].push_back(avalanchePhotons[k]);

                // save new position and momentum of current particle
                particles[i].setMomentum(avalancheParticles[0].getMomentum());
                particles[i].setPosition(avalancheParticles[0].getPosition());
                // push generated particles
                for (int k = 1; k != avalancheParticles.size(); k++)
                    this->generatedParticles[thread_id].push_back(avalancheParticles[k]);

                avalancheParticles.clear();
                timeAvalancheParticles.clear();
                avalanchePhotons.clear();
                timeAvalanchePhotons.clear();
            }
        }

        void RunAvalanche(const FP3& e, const FP3& b, FP timeStep,
            myVector<Particle3d>& avalancheParticles, myVector<FP>& timeAvalancheParticles,
            myVector<Particle3d>& avalanchePhotons, myVector<FP>& timeAvalanchePhotons)
        {
            int countProcessedParticles = 0;
            int countProcessedPhotons = 0;
        
            int avalancheParticlesSize = avalancheParticles.size();
            int avalanchePhotonsSize = avalanchePhotons.size();
        
            while (countProcessedParticles != avalancheParticlesSize
                || countProcessedPhotons != avalanchePhotonsSize)
            {
                for (int k = countProcessedParticles; k != avalancheParticlesSize; k++)
                {
                    oneParticleStep(avalancheParticles[k], e, b, timeAvalancheParticles[k], timeStep,
                        avalanchePhotons, timeAvalanchePhotons);
                }
                avalanchePhotonsSize = avalanchePhotons.size();
                countProcessedParticles = avalancheParticles.size();
        
                for (int k = countProcessedPhotons; k != avalanchePhotonsSize; k++)
                {
                    onePhotonStep(avalanchePhotons[k], e, b, timeAvalanchePhotons[k], timeStep,
                        avalancheParticles, timeAvalancheParticles);
                }
                avalancheParticlesSize = avalancheParticles.size();
                countProcessedPhotons = avalanchePhotons.size();
            }
        }

        void oneParticleStep(Particle3d& particle, const FP3& e, const FP3& b, FP& time, FP timeStep,
            myVector<Particle3d>& avalanchePhotons, myVector<FP>& timeAvalanchePhotons)
        {
            FP invLightVelocity = (FP)1 / Constants<FP>::lightVelocity();
            FP invSchwingerField = (FP)1 / this->schwingerField;
        
            while (time < timeStep)
            {
                FP3 v = particle.getVelocity();
        
                FP H_eff = sqr(e + invLightVelocity * VP(v, b))
                    - sqr(SP(e, v) / Constants<FP>::lightVelocity());
                if (H_eff < 0) H_eff = 0;
                H_eff = sqrt(H_eff);
                FP gamma = particle.getGamma();
                FP chi = gamma * H_eff * invSchwingerField;
                FP rate = (FP)0.0, dt = (FP)2 * timeStep;
                if (chi > 0.0 && this->photonEmissionEnabled)
                {
                    rate = compton.rate(chi);
                    dt = getDt(rate, chi, gamma);
                }
        
                if (dt + time > timeStep)
                {
                    Boris(particle, e, b, timeStep - time);
                    time = timeStep;
                }
                else
                {
                    Boris(particle, e, b, dt);
                    time += dt;
        
                    FP3 v = particle.getVelocity();
                    FP H_eff = sqr(e + invLightVelocity * VP(v, b))
                        - sqr(SP(e, v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    FP gamma = particle.getGamma();
                    FP chi_new = gamma * H_eff * invSchwingerField;
        
                    FP delta = photonGenerator((chi + chi_new) * (FP)0.5);
        
                    Particle3d newPhoton;
                    newPhoton.setType(Photon);
                    newPhoton.setWeight(particle.getWeight());
                    newPhoton.setPosition(particle.getPosition());
                    newPhoton.setMomentum(delta * particle.getMomentum());
        
                    avalanchePhotons.push_back(newPhoton);
                    timeAvalanchePhotons.push_back(time);
                    particle.setMomentum(((FP)1 - delta) * particle.getMomentum());
                }
            }
        }

        void onePhotonStep(Particle3d& photon, const FP3& e, const FP3& b, FP& time, FP timeStep,
            myVector<Particle3d>& avalancheParticles, myVector<FP>& timeAvalancheParticles)
        {
            FP3 k_ = photon.getVelocity();
            k_ = ((FP)1 / k_.norm()) * k_; // normalized wave vector
            FP H_eff = sqrt(sqr(e + VP(k_, b)) - sqr(SP(e, k_)));
            FP gamma = photon.getMomentum().norm()
                / (Constants<FP>::electronMass() * Constants<FP>::lightVelocity());
            FP chi = gamma * H_eff / this->schwingerField;
        
            FP rate = 0.0, dt = (FP)2 * timeStep;
            if (chi > 0.0 && this->pairProductionEnabled)
            {
                rate = breit_wheeler.rate(chi);
                dt = getDt(rate, chi, gamma);
            }
        
            if (dt + time > timeStep)
            {
                photon.setPosition(photon.getPosition()
                    + (timeStep - time) * Constants<FP>::lightVelocity() * k_);
                time = timeStep;
            }
            else
            {
                photon.setPosition(photon.getPosition()
                    + dt * Constants<FP>::lightVelocity() * k_);
                time += dt;
                FP delta = pairGenerator(chi);
        
                Particle3d newParticle;
                newParticle.setType(Electron);
                newParticle.setWeight(photon.getWeight());
                newParticle.setPosition(photon.getPosition());
                newParticle.setMomentum(delta * photon.getMomentum());
        
                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);
        
                newParticle.setType(Positron);
                newParticle.setMomentum(((FP)1 - delta) * photon.getMomentum());
                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);
        
                photon.setP((FP)0.0 * photon.getP());  // mark old photon as deleted
                time = timeStep;
            }
        }

        FP getDt(FP rate, FP chi, FP gamma)
        {
            FP r = -log(random_number_omp());
            r *= gamma / chi;
            return r / rate;
        }

        FP photonGenerator(FP chi)
        {
            FP r = random_number_omp();
            return compton.inv_cdf(r, chi);
        }

        FP pairGenerator(FP chi)
        {
            FP r = random_number_omp();
            if (r < (FP)0.5)
                return breit_wheeler.inv_cdf(r, chi);
            else
                return (FP)1.0 - breit_wheeler.inv_cdf((FP)1.0 - r, chi);
        }

        void operator()(ParticleProxy3d* particle, ValueField field, FP timeStep)
        {}

        void operator()(Particle3d* particle, ValueField field, FP timeStep)
        {
            ParticleProxy3d particleProxy(*particle);
            this->operator()(&particleProxy, field, timeStep);
        }

        Compton compton;
        Breit_wheeler breit_wheeler;

    private:

        FP random_number_omp()
        {
            int thread_id = OMP_GET_THREAD_NUM();
            return distribution[thread_id](randGenerator[thread_id]);
        }

        std::vector<std::default_random_engine> randGenerator;
        std::vector<std::uniform_real_distribution<FP>> distribution;

        std::vector<std::vector<Particle3d>> generatedPhotons, generatedParticles;

        FP schwingerField;

        bool photonEmissionEnabled = true, pairProductionEnabled = true;
    };

    typedef Scalar_Fast_QED_v5<YeeGrid> Scalar_Fast_QED_Yee_v5;
    typedef Scalar_Fast_QED_v5<PSTDGrid> Scalar_Fast_QED_PSTD_v5;
    typedef Scalar_Fast_QED_v5<PSATDGrid> Scalar_Fast_QED_PSATD_v5;
    typedef Scalar_Fast_QED_v5<AnalyticalField> Scalar_Fast_QED_Analytical_v5;


    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////


    template <class TGrid>  // may be AnalyticalField or any Grid type
    class Scalar_Fast_QED_v6 : public ParticlePusher
    {
    public:

        template <class T>
        struct myVector {
            int n = 0, capacity = 10;
            std::vector<T> data;

            myVector() : data(capacity) {}

            T& operator[](int i) { return data[i]; }
            const T& operator[](int i) const { return data[i]; }

            const int& size() const { return n; }
            void clear() { n = 0; }

            void push_back(const T& value) {
                if (n == capacity) {
                    capacity *= 2;
                    data.resize(capacity);
                }
                data[n++] = value;
            }

        };

        Scalar_Fast_QED_v6() : compton(), breit_wheeler()
        {
            this->schwingerField = sqr(Constants<FP>::electronMass() * Constants<FP>::lightVelocity())
                * Constants<FP>::lightVelocity() / (-Constants<FP>::electronCharge() * Constants<FP>::planck());

            int max_threads = OMP_GET_MAX_THREADS();

            randGenerator.resize(max_threads);
            distribution.resize(max_threads);
            for (int i = 0; i < max_threads; i++) {
                randGenerator[i].seed();
                distribution[i] = std::uniform_real_distribution<FP>(0.0, 1.0);
            }

            this->generatedPhotons.resize(max_threads);
            this->generatedParticles.resize(max_threads);
        }

        void disablePhotonEmission()
        {
            this->photonEmissionEnabled = false;
        }

        void enablePhotonEmission()
        {
            this->photonEmissionEnabled = true;
        }

        void disablePairProduction()
        {
            this->pairProductionEnabled = false;
        }

        void enablePairProduction()
        {
            this->pairProductionEnabled = true;
        }

        void processParticles(Ensemble3d* particles,
            const std::vector<std::vector<FP3>>& e, const std::vector<std::vector<FP3>>& b, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            if ((*particles)[Photon].size())
                HandlePhotons((*particles)[Photon], e[0], b[0], timeStep);
            if ((*particles)[Electron].size())
                HandleParticles((*particles)[Electron], e[1], b[1], timeStep);
            if ((*particles)[Positron].size())
                HandleParticles((*particles)[Positron], e[2], b[2], timeStep);

            (*particles)[Photon].clear();

            for (int th = 0; th < max_threads; th++)
            {
                for (int ind = 0; ind < this->generatedPhotons[th].size(); ind++)
                {
                    particles->addParticle(this->generatedPhotons[th][ind]);
                }
                for (int ind = 0; ind < this->generatedParticles[th].size(); ind++)
                {
                    particles->addParticle(this->generatedParticles[th][ind]);
                }

                this->generatedPhotons[th].clear();
                this->generatedParticles[th].clear();
            }
        }

        // rewrite 2 borises
        void Boris(Particle3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void Boris(ParticleProxy3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        void HandlePhotons(ParticleArray3d& photons, const std::vector<FP3>& ve,
            const std::vector<FP3>& vb, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            std::vector<myVector<Particle3d>> avalancheParticlesThread(max_threads);
            std::vector<myVector<FP>> timeAvalancheParticlesThread(max_threads);
            std::vector<myVector<Particle3d>> avalanchePhotonsThread(max_threads);
            std::vector<myVector<FP>> timeAvalanchePhotonsThread(max_threads);

#pragma omp parallel for
            for (int i = 0; i < photons.size(); i++)
            {
                FP3 pPos = photons[i].getPosition();
                FP3 e = ve[i], b = vb[i];

                int thread_id = OMP_GET_THREAD_NUM();

                myVector<Particle3d>& avalancheParticles = avalancheParticlesThread[thread_id];
                myVector<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[thread_id];
                myVector<Particle3d>& avalanchePhotons = avalanchePhotonsThread[thread_id];
                myVector<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[thread_id];

                // start avalanche with current photon
                avalanchePhotons.push_back(photons[i]);
                timeAvalanchePhotons.push_back((FP)0.0);

                RunAvalanche(e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push generated physical photons
                for (int k = 0; k != avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0)
                        this->generatedPhotons[thread_id].push_back(avalanchePhotons[k]);

                // push generated particles
                for (int k = 0; k != avalancheParticles.size(); k++)
                    this->generatedParticles[thread_id].push_back(avalancheParticles[k]);

                avalancheParticles.clear();
                timeAvalancheParticles.clear();
                avalanchePhotons.clear();
                timeAvalanchePhotons.clear();
            }
        }

        void HandleParticles(ParticleArray3d& particles, const std::vector<FP3>& ve,
            const std::vector<FP3>& vb, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            std::vector<myVector<Particle3d>> avalancheParticlesThread(max_threads);
            std::vector<myVector<FP>> timeAvalancheParticlesThread(max_threads);
            std::vector<myVector<Particle3d>> avalanchePhotonsThread(max_threads);
            std::vector<myVector<FP>> timeAvalanchePhotonsThread(max_threads);

#pragma omp parallel for
            for (int i = 0; i < particles.size(); i++)
            {
                FP3 pPos = particles[i].getPosition();
                FP3 e = ve[i], b = vb[i];

                int thread_id = OMP_GET_THREAD_NUM();

                myVector<Particle3d>& avalancheParticles = avalancheParticlesThread[thread_id];
                myVector<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[thread_id];
                myVector<Particle3d>& avalanchePhotons = avalanchePhotonsThread[thread_id];
                myVector<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[thread_id];

                // start avalanche with current particle
                avalancheParticles.push_back(particles[i]);
                timeAvalancheParticles.push_back((FP)0.0);

                RunAvalanche(e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push physical photons
                for (int k = 0; k != avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0) // include "removePhotonsBelow"
                        this->generatedPhotons[thread_id].push_back(avalanchePhotons[k]);

                // save new position and momentum of current particle
                particles[i].setMomentum(avalancheParticles[0].getMomentum());
                particles[i].setPosition(avalancheParticles[0].getPosition());
                // push generated particles
                for (int k = 1; k != avalancheParticles.size(); k++)
                    this->generatedParticles[thread_id].push_back(avalancheParticles[k]);

                avalancheParticles.clear();
                timeAvalancheParticles.clear();
                avalanchePhotons.clear();
                timeAvalanchePhotons.clear();
            }
        }

        void RunAvalanche(const FP3& e, const FP3& b, FP timeStep,
            myVector<Particle3d>& avalancheParticles, myVector<FP>& timeAvalancheParticles,
            myVector<Particle3d>& avalanchePhotons, myVector<FP>& timeAvalanchePhotons)
        {
            int countProcessedParticles = 0;
            int countProcessedPhotons = 0;

            int avalancheParticlesSize = avalancheParticles.size();
            int avalanchePhotonsSize = avalanchePhotons.size();

            while (countProcessedParticles != avalancheParticlesSize)
            {
                for (int k = countProcessedParticles; k != avalancheParticlesSize; k++)
                {
                    oneParticleStep(avalancheParticles[k], e, b, timeAvalancheParticles[k], timeStep,
                        avalanchePhotons, timeAvalanchePhotons);
                }
                avalanchePhotonsSize = avalanchePhotons.size();
                countProcessedParticles = avalancheParticlesSize;

                for (int k = countProcessedPhotons; k != avalanchePhotonsSize; k++)
                {
                    onePhotonStep(avalanchePhotons[k], e, b, timeAvalanchePhotons[k], timeStep,
                        avalancheParticles, timeAvalancheParticles);
                }
                avalancheParticlesSize = avalancheParticles.size();
                countProcessedPhotons = avalanchePhotonsSize;
            }
        }

        void oneParticleStep(Particle3d& particle, const FP3& e, const FP3& b, FP& time, FP timeStep,
            myVector<Particle3d>& avalanchePhotons, myVector<FP>& timeAvalanchePhotons)
        {
            FP invLightVelocity = (FP)1 / Constants<FP>::lightVelocity();
            FP invSchwingerField = (FP)1 / this->schwingerField;

            while (time < timeStep)
            {
                FP3 v = particle.getVelocity();

                FP H_eff = sqr(e + invLightVelocity * VP(v, b))
                    - sqr(SP(e, v) / Constants<FP>::lightVelocity());
                if (H_eff < 0) H_eff = 0;
                H_eff = sqrt(H_eff);
                FP gamma = particle.getGamma();
                FP chi = gamma * H_eff * invSchwingerField;
                FP rate = (FP)0.0, dt = (FP)2 * timeStep;
                if (chi > 0.0 && this->photonEmissionEnabled)
                {
                    rate = compton.rate(chi);
                    dt = getDt(rate, chi, gamma);
                }

                if (dt + time > timeStep)
                {
                    Boris(particle, e, b, timeStep - time);
                    time = timeStep;
                }
                else
                {
                    Boris(particle, e, b, dt);
                    time += dt;

                    FP3 v = particle.getVelocity();
                    FP H_eff = sqr(e + invLightVelocity * VP(v, b))
                        - sqr(SP(e, v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    FP gamma = particle.getGamma();
                    FP chi_new = gamma * H_eff * invSchwingerField;

                    FP delta = photonGenerator((chi + chi_new) * (FP)0.5);

                    Particle3d newPhoton;
                    newPhoton.setType(Photon);
                    newPhoton.setWeight(particle.getWeight());
                    newPhoton.setPosition(particle.getPosition());
                    newPhoton.setMomentum(delta * particle.getMomentum());

                    avalanchePhotons.push_back(newPhoton);
                    timeAvalanchePhotons.push_back(time);
                    particle.setMomentum(((FP)1 - delta) * particle.getMomentum());
                }
            }
        }

        void onePhotonStep(Particle3d& photon, const FP3& e, const FP3& b, FP& time, FP timeStep,
            myVector<Particle3d>& avalancheParticles, myVector<FP>& timeAvalancheParticles)
        {
            FP3 k_ = photon.getVelocity();
            k_ = ((FP)1 / k_.norm()) * k_; // normalized wave vector
            FP H_eff = sqrt(sqr(e + VP(k_, b)) - sqr(SP(e, k_)));
            FP gamma = photon.getMomentum().norm()
                / (Constants<FP>::electronMass() * Constants<FP>::lightVelocity());
            FP chi = gamma * H_eff / this->schwingerField;

            FP rate = 0.0, dt = (FP)2 * timeStep;
            if (chi > 0.0 && this->pairProductionEnabled)
            {
                rate = breit_wheeler.rate(chi);
                dt = getDt(rate, chi, gamma);
            }

            if (dt + time > timeStep)
            {
                photon.setPosition(photon.getPosition()
                    + (timeStep - time) * Constants<FP>::lightVelocity() * k_);
                time = timeStep;
            }
            else
            {
                photon.setPosition(photon.getPosition()
                    + dt * Constants<FP>::lightVelocity() * k_);
                time += dt;
                FP delta = pairGenerator(chi);

                Particle3d newParticle;
                newParticle.setType(Electron);
                newParticle.setWeight(photon.getWeight());
                newParticle.setPosition(photon.getPosition());
                newParticle.setMomentum(delta * photon.getMomentum());

                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);

                newParticle.setType(Positron);
                newParticle.setMomentum(((FP)1 - delta) * photon.getMomentum());
                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);

                photon.setP((FP)0.0 * photon.getP());  // mark old photon as deleted
                time = timeStep;
            }
        }

        FP getDt(FP rate, FP chi, FP gamma)
        {
            FP r = -log(random_number_omp());
            r *= gamma / chi;
            return r / rate;
        }

        FP photonGenerator(FP chi)
        {
            FP r = random_number_omp();
            return compton.inv_cdf(r, chi);
        }

        FP pairGenerator(FP chi)
        {
            FP r = random_number_omp();
            if (r < (FP)0.5)
                return breit_wheeler.inv_cdf(r, chi);
            else
                return (FP)1.0 - breit_wheeler.inv_cdf((FP)1.0 - r, chi);
        }

        void operator()(ParticleProxy3d* particle, ValueField field, FP timeStep)
        {}

        void operator()(Particle3d* particle, ValueField field, FP timeStep)
        {
            ParticleProxy3d particleProxy(*particle);
            this->operator()(&particleProxy, field, timeStep);
        }

        Compton compton;
        Breit_wheeler breit_wheeler;

    private:

        FP random_number_omp()
        {
            int thread_id = OMP_GET_THREAD_NUM();
            return distribution[thread_id](randGenerator[thread_id]);
        }

        std::vector<std::default_random_engine> randGenerator;
        std::vector<std::uniform_real_distribution<FP>> distribution;

        std::vector<std::vector<Particle3d>> generatedPhotons, generatedParticles;

        FP schwingerField;

        bool photonEmissionEnabled = true, pairProductionEnabled = true;
    };

    typedef Scalar_Fast_QED_v6<YeeGrid> Scalar_Fast_QED_Yee_v6;
    typedef Scalar_Fast_QED_v6<PSTDGrid> Scalar_Fast_QED_PSTD_v6;
    typedef Scalar_Fast_QED_v6<PSATDGrid> Scalar_Fast_QED_PSATD_v6;
    typedef Scalar_Fast_QED_v6<AnalyticalField> Scalar_Fast_QED_Analytical_v6;
}
