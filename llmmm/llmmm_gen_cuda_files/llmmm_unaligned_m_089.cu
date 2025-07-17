#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_89_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<89> mm_instantiator;

public:
  UnalignedM_89_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_89_MMInstantiatorWrapper__;

}  // namespace LLMMM
